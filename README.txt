CS364 - AI - Final Project
Katja Zoner, Declan Galleher, Rachel Clark, Max Kramer

This directory contains:

- LaneDetector.py: Our final version of the lane detection program. (Try out this one!)
This script performs all the image processing and calculations to detect single 
Left and Right lane lines and report the car's distance from the center of the lane.
Functions in this script are called by the car_main.py program to get lane data for the
RC car, however, this script can also be run on its own as: python3 LaneDetector.py <optional fileName>.
When run without a file name given on the command line, several video and image files will be suggested
to the user. When run in the standalone version, image files will be presented on the screen following processing,
whereas video files will be processed and saved into the folder video_output.

- test_images: A folder containing several road images for testing purposes.

- test_video: A folder containing several road video clips for testing purposes.

- video_output: A folder containing the processed video files. Newly processed videos
are written to this folder.

- image_output: A folder containing several images from different stages in the image 
processing pipeline (these images were used in our PPT).

- motor_control.py: A script that contains basic movement functions (forward, reverse, turn L, turn R, pivot L, pivot R)
 for the car. **Note: Pin numbers in script no longer match pinouts in car, as car wiring has changed as we are trying to fix it.**

- dist_sensor.py: A script that contains functions for checking distance sensors. frontDist() returns distance of object closest
to the front of the car as detected by the supersonic sensor.**Note: Pin numbers in script no longer match pinouts in car, as car 
wiring has changed as we are trying to fix it.**
 
- car_main.py: The main script for controlling the RC car. Contains functions to check front for obstacles and make a set number of 
attempts to cirvumnavigate obstacle (before stopping and waiting for help), as well as an autodrive function to drive and make gentle turns,
while checking front for obstacles. Time permitted, this script would use the functions in the lane detection script to get the car's distance 
from the center of the lane and adjust the car's course accordingly. We were unable to add these features as further testing with the car was
delayed by technical issues with the motor drivers.

- several older test versions of the lane detection algorithm


