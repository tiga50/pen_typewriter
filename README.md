# Pen TypeWriter

### Overview

This is my first project of developing a Deep Learning application. The [TensorFlow Microcontroller Challenge](https://experiments.withgoogle.com/tfmicrochallenge)  motivated me to start this project. After pondering an application with the combination of [Arduino Sense 33 BLE](https://store.arduino.cc/usa/nano-33-ble-sense "Arduino Store") and [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers "TFL4M"), I came up with an idea of Pen TypeWriter that recognizes the motion of pen writing and output a recognized character via USB-Keyboard I/F. I have developed the initial implementation of Pen TypeWriter. Though its recognition is not so accurate, I want to share my experience of this development. I wish I have such tiny pen which can be used as key input method of smartphones and a tablet in the future.

--- 
### Project Description

This project is to develop a Deep Leaning application using the [Arduino Sense 33 BLE](https://store.arduino.cc/usa/nano-33-ble-sense "Arduino Store").  Arduino Sense 33 BLE is an AI-enabled circuit board that embeds a microprocessor and various sensors in a small form factor. Thanks to [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers "TFL4M"), it enables to analysis output of sensors with the most popular deep learning technology. The application of this project, Pen Typewriter, is to recognize the motion of pen writing by attaching the Arduino Sensor 33 BLE at the top of the pen and output a recognized alphabet and number character via USB-keyboard I/F. The following video shows how it works.

https://user-images.githubusercontent.com/87429220/125905180-c38b5eda-bfe0-40da-a31e-fc63886a1ef8.MOV----


There are many examples to recognize hand gestures using motion sensors on Arudiono Sense 33 BLE. For example, [Magic Wand](https://create.arduino.cc/projecthub/user1382012/magic-wand-29fa3b). However, recognizing the motion of pen writing has the following issues.

- The motion of pen writing is small compared to motions of hand gesture
- The time of pen motion is various depending on a writing character 
 
There are also other issues such as the following but they are out of the scope of this project.

- Size of characters (figure&wrist motion v.s. arm motion)
- Individual differences in writing style
- Continuous writing of multiple characters
- International character sets, e.g. Hiragana and Kanji in Japanese

This project has explored only the case of writing a single alphabet and number character with a fixed size(figure&palm motion). The result of the exploration is as follows.

- Improved the detection of start and end of motion
- Made sampling stable  by indicating the initial pen position by the LEDs 
- Can not achieve accurate recognition in both number and alphabet, though there would be room to improve DNN network
- Can support the recognition of the number characters but can not support the alphabet characters due to the limitation of the Flash ROM size

--- 
### Instruction
#### Install and Run

1. Install the [Arduino IDE ], build the sketch [pen_typewriter.ino](arduino/pen_typewriter.ino) and flash it on the Arduino Nano Sense 33 BLE
2. Attache the Arudino Nano at the top of a pen as in the above movie. Note that the right orientation of the Arduino Nano is important
2. Plug the Arduino Nano Sense 33 BLE into the USB port of a PC or Mac. The Arduino Nano will be recognized as a USB keyboard
3. Open any kind of text editing application, e.g. Notepad on PC or TextEdit on Mac
4. Hand the pen with the Arduino Nano with the right hand, then the red LED on the Arduino Nano will be flashing
5. Stand the pen at the writing position, then the green LED will be flashing
6. Hold the pen position until the green LED turns on
7. Start writing a number character (0-9) with a size (around 2 inches) by moving your figure & wrist
8. The recognized character will be input on the application

#### Train your DNN model and Run

1. Comment out "#define ONLY_CAPTURING" and "#define USB_KEYBOARD" in the [pen_typewriter.ino](arduino/pen_typewriter.ino). This makes the Andino Nano output sample values via the serial port. Please use the 'SerialMonitor' of Arduino IDE to see the output
2. Capture some training data by writing characters and train your DNN model. Please refer to [TensorFlow Lite for Microcontrollers]( https://www.tensorflow.org/lite/microcontrollers)
3. Replace [model.h](arduino/model.h) with your DNN model
4. Uncomment "#define ONLY_CAPTURING" in the [pen_typewriter.ino](arduino/pen_typewriter.ino). If you want to output the result of recognition via the serial port, comment out "#define USB_KEYBOARD"
5. Build the sketch, flash it and run as above

The following picture shows how I captured the training data of number characters for training my DNN model.
![SamplingTrainingData](https://user-images.githubusercontent.com/87429220/125924344-827329bf-94c9-45c9-9be6-0b4b7b4bfbea.jpg)

Good Luck!
