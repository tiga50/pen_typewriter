/*  PenTypeWriter

  This code uses Tensorflow Lite for Microcontrollers to recognize 
  the motion of pen writing by attaching the Arduino Sensor 33 BLE 
  at the top of the pen. A recognized alphabet and number character
  will be output via USB-keyboard I/F.

  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.

  Created by Tatsuya Igarashi

  This code is in the public domain.
*/

#include <Arduino_LSM9DS1.h>
#include "Arduino.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <PluggableUSBHID.h>
#include <USBKeyboard.h>

#define USB_KEYBOARD  // the result is used as usb keyboard
// #define ONLY_CAPTURING  // enable only when capturing data for DNN training
// #define NORMALIZE_SAMPLE_VALUE  // enable this in the case the DNN model requires the normalization
//#define DEBUG_SAMPLING   // enable to the detial of sampling at the SerialPlotter
//#define DEBUG_PEN   // enable to output the verbose debug messages
// #define DEBUG_INPUT_TENSOR

#ifdef USB_KEYBOARD
USBKeyboard keyboard;
#endif

#ifdef DATA_OF_ABC
#include "data4/model_256_26_2.h"
const char* GESTURES[] = {
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
  "U", "V", "W", "X", "Y","Z"
};
#endif
#define DATA_of_NUMBERS
#ifdef DATA_of_NUMBERS
// #include "data5/model_data5_256_128_3_10.h"
#include "model.h"
// array to map gesture index to a name depending on the DNN model

const char* GESTURES[] = {
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
  //  "0", "1", "2"
};
#endif

#ifdef ONLY_CAPTURING
const int numParams = 6;  // aX, aY, aZ, gX, gY, gZ
#else  // the following depends on the DNN model
//const int numParams = 6; // aX, aY, aZ, gX, gY, gZ
//const int numParams = 4;  // aX, aY, aZ, aS
const int numParams = 3;  // aX, aY, aZ
//const int numParams = 2;  // aY, aZ
#endif

#ifdef ONLY_CAPTURING
const int maxSamples = 512;
#else // The following depends on the DNN model
// const int maxSamples = 512;  // NG data 3 NumParams3, numGestures=3 Model is 3252704 bytes model.h, is 20,058,376 bytes.region `FLASH' overflowed by 2518584 bytes
// const int maxSamples = 512;  // NG daata4 NumParam=3, numGestures=26, Model is 3257408 bytes model.h, is 20,087,384 bytes.region `FLASH' overflowed by 2523296 bytes
// const int maxSamples = 256;     // OK data3 MnumParams=3,numGestures=3 Model is 303128 bytes, model.h, is 1,869,324 bytes. Sketch uses 555792 bytes (56%) of program storage space. Maximum is 983040 bytes
// const int maxSamples = 256;     // NG data4 NummParams=3, numGesture=26 Model is 845836 bytes model.h, is 5,216,024 bytes., Region `FLASH' overflowed by 111712 bytes
// const int maxSamples = 256;     // NG data4 numGesture=3  MnumParams=3, Model is 841204 bytes model.h, is 5,187,460 bytes. region `FLASH' overflowed by 107080 bytes
const int maxSamples = 256;     // OK data4 numGesture=26 MnumParams=2 (aY, aZ),  Model is 583732 bytes model.h, is 3,599,716 bytes.Sketch uses 836384 bytes (85%) of program storage space. Maximum is 983040 bytes.
// const int maxSamples = 256;     // OK data4  numGesture3, numParams=2,, Model is 579060 bytes, model.h, is 3,570,904 bytes.Sketch uses 831552 bytes (84%) of program storage space. Maximum is 983040 bytes.
#endif


const int sensingReadyCount = 200;      // counting while the acceration  scalar is less than sensingAccerationTreshould
const float sensingAccelerationTreshold = 0.1;
const float startAccelerationThreshold = 0.2; // threshold of significant in G's
const int startAccelerationCount = 2;
const float endAccelerationThreshold = 0.2; // threshold of significant in G's
const int endAccelerationCount = 25;
const int restartPresampingCount = 40;
const float penPosition_aXmin = -0.9;
const float penPosition_aXmax = -0.6;
const float penPosition_aYmin = 0.0;
const float penPosition_aYmax = 0.6;
const float penPosition_aZmin = 0.0;
const float penPosition_aZmax = 0.4;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 48 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));


struct Sample {
  float aS, aStart, aThreshold;
  float aX, aY, aZ, gX, gY, gZ;
};

float aXn, aYn, aZn, gXn, gYn, gZn;

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
#ifndef USB_KEYBOARD
  Serial.begin(9600);
  while (!Serial);
#endif

  initLed();

  // initialize the IMU
  if (!IMU.begin()) {
#ifndef  USB_KEYBOARD
    Serial.println("Failed to initialize IMU!");
#endif
    while (1);
  }

#ifndef ONLY_CAPTURING
  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");
  Serial.println();
#else
#ifndef USB_KEYBOARD
  // print the header
#ifdef DEBUG_SAMPLING
  Serial.println("aX,aY,aZ,aS,aStart,aTreshold");
#else
  Serial.println("aX,aY,aZ,gX,gY,gZ");
#endif
#endif
#endif

#ifndef ONLY_CAPTURING
  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
#ifndef USB_KEYBOARD
    Serial.println("Model schema mismatch!");
#endif
    while (1);
  }
#ifndef USB_KEYBOARD
  Serial.print("TFL version = ");
  Serial.print(tflModel->version());
  Serial.println();
#endif

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
#endif

  // initalize acumunated samples
  aXn = aYn = aZn = 0.0;
  gXn = gXn = gXn = 0.0;
}


boolean getPenSample(bool &penPositionOK,
                     float &aS,
                     float &aX, float &aY, float &aZ,
                     float &gX, float &gY, float &gZ) {

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float bias = 0.90;
    IMU.readAcceleration(aX, aY, aZ);
    IMU.readGyroscope(gX, gY, gZ);

    penPositionOK = (aX >  penPosition_aXmin && aX <  penPosition_aXmax &&
                     aY > penPosition_aYmin && aY <  penPosition_aYmax &&
                     aZ > penPosition_aZmin && aZ < penPosition_aZmax ) ? true : false;


#ifdef DEBUG_PENPOSITON
    if (!penPositionOK) {
      Serial.print(aX, 3);
      Serial.print(',');
      Serial.print(aY, 3);
      Serial.print(',');
      Serial.print(aZ, 3);
      Serial.println();
    }
#endif

    // lowpath filter
    aXn = aXn * bias +  aX * (1 - bias);
    aYn = aYn * bias +  aY * (1 - bias);
    aZn = aZn * bias +  aZ * (1 - bias);
    gXn = gXn * bias +  gX * (1 - bias);
    gYn = gYn * bias +  gY * (1 - bias);
    gZn = gZn * bias +  gZ * (1 - bias);

    // remove gravity
    aX -= aXn;
    aY -= aYn;
    aZ -= aZn;

    // caliculate a vector of the accelerator values
    aS = sqrt(pow(aX, 2.0) + pow(aY, 2.0) + pow(aZ, 2.0));

    return  true;
  }
  return false;
}

void printPenSample(float aS,
                    float aStart, float aTreshold,
                    float aX, float aY, float aZ,
                    float gX, float gY, float gZ) {

  // print the data in CSV format
#ifdef DEBUG_SAMPLING
  Serial.print(aX - 0, 5);
  Serial.print(',');
  Serial.print(aY - 1.5, 3);
  Serial.print(',');
  Serial.print(aZ - 3, 1);
  Serial.print(',');

  Serial.print(aStart + 3, 3);
  Serial.print(',');
  Serial.print(aTreshold + 3, 3);
  Serial.print(',');
  Serial.print(aS + 3, 3);
#else
  Serial.print(aX, 3);
  Serial.print(',');
  Serial.print(aY, 3);
  Serial.print(',');
  Serial.print(aZ, 3);
  Serial.print(',');
  Serial.print(gX, 3);
  Serial.print(',');
  Serial.print(gY, 3);
  Serial.print(',');
  Serial.print(gZ, 3);
#endif

  Serial.println();
}

static bool ledInitalized = false;
void initLed() {
  // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  ledInitalized = true;
}

void setLed(int led, bool flash) {
  static int currentLed = LEDR;
  static int count = 0;
  static bool on = true;

  if (!ledInitalized)
    return;

  if (led != currentLed) {
    digitalWrite(currentLed, HIGH);
    digitalWrite(led, LOW);   // on
    currentLed = led;
    // Serial.print("led=");
    // Serial.println(led);
    count = 0;
    on = true;
  } else {
    if (!flash) {
      if (!on)
        count = 40;   // make led on on
    } else {
      count++;
    }
    if (count == 20) {
      digitalWrite(currentLed, HIGH); // off
      on = false;
    } else if (count == 40) {
      digitalWrite(currentLed, LOW);  // on
      count = 0;
      on = true;
    }
  }
}

int getSamples(Sample* samples) {
  float aS;
  float aX, aY, aZ, gX, gY, gZ;
  float aStart;
  int numSamples;
  int aCount;
  bool penPositionOK;
  int stayCount;

  // wait until PenPosition becomes stable and then moving fast
  stayCount = 0;
  while (true) {
    if (!getPenSample(penPositionOK, aS, aX, aY, aZ, gX, gY, gZ)) {
      continue;
    }
    if (penPositionOK) {
      if (stayCount >= sensingReadyCount) {
        setLed(LEDG, false); // pen is now stable light  GREEN
        break;
      } else {
        stayCount++;
        setLed(LEDG, true);    // sampling, flashing GREE
      }
    } else {
      setLed(LEDR, true);  // penPosition is not OK, flashing RED
      stayCount = 0;
    }
  }

#ifdef DEBUG_PEN
  Serial.println("# Pen becomes stable");
#endif

  // wait until  moving fast
  while (true) {
    if (!getPenSample(penPositionOK, aS, aX, aY, aZ, gX, gY, gZ)) {
      continue;
    }
    if (aS > sensingAccelerationTreshold)
      break;
  }


#ifdef DEBUG_PEN
  Serial.println("# Start pre-sampling");
#endif

  // start presampling
  numSamples = 0;
  aCount = 0;
  aStart = 0.0;
  setLed(LEDB, true);    // presampling, light on BLUE

  while (numSamples < maxSamples) {
    if (!getPenSample(penPositionOK, aS, aX, aY, aZ, gX, gY, gZ)) {
      continue;
    }
    if ( aS >= startAccelerationThreshold) {
      if (++aCount > startAccelerationCount) {
        aStart = aS;
        break;
      }
    } else { // aS < startAccelerationTrethould
      if (numSamples == restartPresampingCount) {
        // aS has been not betond startAcceleationThreshold during restartResamplingCount
#ifdef DEBUG_PEN
        Serial.print("# RestartPresampling sampling numSamples=");
        Serial.println(numSamples);
#endif
        return 0;   // starting over
      }
    }
    samples[numSamples].aS = aS;
    samples[numSamples].aStart = aStart;
    samples[numSamples].aThreshold = aCount == 0 ? 0.0 : startAccelerationThreshold;
    samples[numSamples].aX = aX;
    samples[numSamples].aY = aY;
    samples[numSamples].aZ = aZ;
    samples[numSamples].gX = gX;
    samples[numSamples].gY = gY;
    samples[numSamples].gZ = gZ;
    numSamples ++;
  }

#ifdef DEBUG_PEN
  Serial.print("# start sampling numSamples=");
  Serial.println(numSamples);
#endif
  // continue sampling while aS has been less than endAccelertionThread during endAccertionCount
  aCount = 0;
  setLed(LEDB, false);    // sampling, light on BLUE

  while (numSamples < maxSamples) {
    if (!getPenSample(penPositionOK, aS, aX, aY, aZ, gX, gY, gZ)) {
      continue;
    }
    if (aS < endAccelerationThreshold) {
      if (++aCount > endAccelerationCount) {
#ifdef DEBUG_PEN
        Serial.print("# end sampling  aCount=");
        Serial.println(aCount);
#endif
        break;            // stopSamping
      }
    } else { // aS >= endAccelerationTreshould
      // reset counting
#ifdef DEBUG_PEN
      if (aCount > 0) {
        Serial.print("# Reset the end counting aCount=");
        Serial.println(aCount);
      }
#endif
      aCount = 0;
    }
    samples[numSamples].aS = aS;
    samples[numSamples].aStart = aStart;
    samples[numSamples].aThreshold = aCount == 0 ? 0.0 : endAccelerationThreshold;
    samples[numSamples].aX = aX;
    samples[numSamples].aY = aY;
    samples[numSamples].aZ = aZ;
    samples[numSamples].gX = gX;
    samples[numSamples].gY = gY;
    samples[numSamples].gZ = gZ;
    numSamples ++;
  }
#ifdef DEBUG_PEN
  Serial.print("# Stop Sampling numSamples=");
  Serial.println(numSamples);
#endif
  setLed(LEDR, false);
  return numSamples;
}

void loop() {
  static int numSamples;
  static Sample Samples[maxSamples];

  while ((numSamples = getSamples(Samples)) <= endAccelerationCount)
    ;


#ifdef ONLY_CAPTURING
  for (int i = 0; i < numSamples; i++) {
    Sample sample = Samples[i];
    printPenSample(sample.aS, sample.aStart, sample.aThreshold,
                   sample.aX, sample.aY, sample.aZ, sample.gX, sample.gY, sample.gZ);
  }
#ifdef DEBUG_SAMPLING
  for (int i = 0; i < 30; i++) {
    printPenSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }
  Serial.print("# numSamples=");
  Serial.println(numSamples);
#endif

  Serial.println();
  return;
#endif

  /*
    Serial.print("#Inferencing numSamples = ");
    Serial.print(numSamples);
    Serial.print(" dataPos = ");
    Serial.print(dataPos);
    Serial.println();
  */

#ifdef NORMALIZE_SAMPLE_VALUE
  float aMax = 0.0;
  float gMax = 0.0;
  for (int i = 0; i < numSamples; i++) {
    Sample sample = Samples[i];
    if (numParams == 2) {
      float value;
      value = abs(sample.aY);
      if (value > aMax)
        aMax = value;
      value = abs(sample.aZ);
      if (value > aMax)
        aMax = value;
    } else {
      float value;
      value = abs(sample.aX);
      if (value > aMax)
        aMax = value;
      value = abs(sample.aY);
      if (value > aMax)
        aMax = value;
      value = abs(sample.aZ);
      if (value > aMax)
        aMax = value;
    }

    if (numParams == 6) {
      float value;
      value = abs(sample.gX);
      if (value > gMax)
        gMax = value;
      value = abs(sample.gY);
      if (value > gMax)
        gMax = value;
      value = abs(sample.gZ);
      if (value > gMax)
        gMax = value;
    }
  }
  float aRatio = 4.0 / aMax;
  float gRatio = 2000.0 / gMax;
#else
  float aRatio = 1;
  float gRatio = 1;
#endif
  // Serial.print("aMax=");
  // Serial.print(aMax);
  // Serial.print(" aRatio=");
  // Serial.println(aRatio);
  int dataPos = 0;
  for (int i = 0; i < numSamples; i++) {
    Sample sample = Samples[i];
    // normalize the IMU data between 0 to 1 and store in the model's
    // input tensor
    if (numParams == 2) {
      tflInputTensor->data.f[dataPos++] = (sample.aY * aRatio + 4.0) / 8.0;
      tflInputTensor->data.f[dataPos++] = (sample.aZ * aRatio + 4.0) / 8.0;
    } else {
      tflInputTensor->data.f[dataPos++] = (sample.aX * aRatio + 4.0) / 8.0;
      tflInputTensor->data.f[dataPos++] = (sample.aY * aRatio + 4.0) / 8.0;
      tflInputTensor->data.f[dataPos++] = (sample.aZ * aRatio + 4.0) / 8.0;
    }
     if (numParams == 4) {
        // tflInputTensor->data.f[dataPos++] = (sample.aS * aRatio + 4.0) / 8.0;
        tflInputTensor->data.f[dataPos++] = sample.aS/4.0;
    }
    if (numParams == 6) {
      tflInputTensor->data.f[dataPos++] = (sample.gX * gRatio + 2000.0) / 4000.0;
      tflInputTensor->data.f[dataPos++] = (sample.gY * gRatio + 2000.0) / 4000.0;
      tflInputTensor->data.f[dataPos++] = (sample.gZ * gRatio + 2000.0) / 4000.0;
    }
  }


#ifdef DEBUG_INPUT_TENSOR
  for (int i = 0; i < numSamples; i++) {
    if (numParams == 2) {
      printPenSample(0, 0, 0,
                     0,
                     tflInputTensor->data.f[i * numParams + 0],
                     tflInputTensor->data.f[i * numParams + 1],
                     0, 0, 0);
    } else if (numParams == 3) {
      printPenSample(0, 0, 0,
                     tflInputTensor->data.f[i * numParams + 0],
                     tflInputTensor->data.f[i * numParams + 1],
                     tflInputTensor->data.f[i * numParams + 2],
                     0, 0, 0);
    } else if (numParams == 4) {
      printPenSample(0, 0, 0,
                     tflInputTensor->data.f[i * numParams + 0],
                     tflInputTensor->data.f[i * numParams + 1],
                     tflInputTensor->data.f[i * numParams + 2],
                     tflInputTensor->data.f[i * numParams + 3],
                     0, 0);
    } else {
      printPenSample(0, 0, 0,
                     tflInputTensor->data.f[i * numParams + 0],
                     tflInputTensor->data.f[i * numParams + 1],
                     tflInputTensor->data.f[i * numParams + 2],
                     tflInputTensor->data.f[i * numParams + 3],
                     tflInputTensor->data.f[i * numParams + 4],
                     tflInputTensor->data.f[i * numParams + 5]);
    }
  }
#endif

  for (int i = numSamples; i < maxSamples; i++) {
    tflInputTensor->data.f[dataPos++] = 0.0;
    tflInputTensor->data.f[dataPos++] = 0.0;
    if (numParams >= 3)
      tflInputTensor->data.f[dataPos++] = 0.0;
    if (numParams == 4)
      tflInputTensor->data.f[dataPos++] = 0.0;
    else if (numParams == 6) {
      tflInputTensor->data.f[dataPos++] = 0.0;
      tflInputTensor->data.f[dataPos++] = 0.0;
      tflInputTensor->data.f[dataPos++] = 0.0;
    }
  }

#ifndef USB_KEYBOARD
  Serial.print("numSamples =");
  Serial.print(numSamples);
  Serial.print(" numParams =");
  Serial.print(numParams);
  Serial.print(" !inputTensor data ");
  Serial.print(tflInputTensor->data.f[0]);
  Serial.print(" dataPos = ");
  Serial.print(dataPos);
  Serial.println();
#endif

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
#ifndef USB_KEYBOARD
    Serial.println("Invoke failed!");
#endif
    while (1);
    return;
  }

  // Loop through the output tensor values from the model
  float maxValue = 0.0;
  int highIdx = 0;
  for (int i = 0; i < NUM_GESTURES; i++) {
#ifndef USB_KEYBOARD
    Serial.print("# ");
    Serial.print(GESTURES[i]);
    Serial.print(": ");
    Serial.println(tflOutputTensor->data.f[i], 6);
#endif
    if (tflOutputTensor->data.f[i] > maxValue) {
      maxValue = tflOutputTensor->data.f[i];
      highIdx = i;
    }
  }
#ifndef USB_KEYBOARD
  Serial.println();
#endif

#ifdef USB_KEYBOARD
  String gesture = GESTURES[highIdx];
  for (int i=0; i < gesture.length(); i++) 
    keyboard.key_code(gesture.charAt(i));
#endif
}
