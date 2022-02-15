# CargoVision2022
 Raspberry Pi and Neural Network Code for Cargo Vision
 
 This code base is an attempt to use the new WPILib Axon tool to detect the red and blue
 cargo game pieces in the 2022 FRC game Rapid React.
 
 The first attempt to train the neural network using just video of the red and blue cargo
 balls did not work well. Inferencing on the Raspberry Pi could not discern between the
 two colored cargo balls.
 
 The latest attempt is to train the model using Google's Open Dataset's "Ball" class images.
 This seems to detect the cargo pieces fairly well but addtional OpenCV processing is necessary
 using red and blue masks to determine whether a ball is actually a red cargo or blue cargo game 
 piece. More work needs to be done to remove false detections.
 
 The two files in the repo are uploaded using the WPILibPi dashboard. The BallTest-0.tar.gz file
 is uploaded via the File Upload section on the Application tab of the dashboard. The upload.py
 file is uploaded on the Application tab in the Vision Application Configuration section.
