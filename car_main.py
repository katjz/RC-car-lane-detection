import RPi.GPIO as gpio
import time
import sys
import random
#import Tkinter as tk

import motor_control as mc
from dist_sensor import *

OBSTACLE_DIST = 25 # Distance at which front sensor reports an obstacle

def init():
    gpio.setmode(gpio.BOARD)
    gpio.setup(7, gpio.OUT)
    gpio.setup(11, gpio.OUT)
    gpio.setup(13, gpio.OUT)
    gpio.setup(15, gpio.OUT)

### Function to check front distance sensors for potential obstacle
#  Returns True if obstacle detected, False if path is clear
def frontCheck():
    init()
    dist = frontDist()

    if dist < OBSTACLE_DIST:
        print("Detected obstacle in front. Distance:",dist)
        #return True
        frontAvoid()
    #return False

### Function to try to avoid obstacle detected in front
# TODO:  - improve to decide whether to turn L or R to avoid
#        - improve to recorrect path after avoiding obstacle
def frontAvoid():
    stuck = True
    attempts = 0
    maxAttempts = 5

    # Reverse and pivot left to try to circumnavigate obstacle
    while stuck:
        init()
        mc.reverse(2)
        init()
        mc.pivotLeft(3)
        dist = frontDist()
        if dist > OBSTACLE_DIST:
            stuck = False
        else:
            attempts+=1
        # Give up after maxAttempts
        if attempts >= maxAttempts:
            print("Cannot avoid obstacle. Waiting for help.")
            sys.exit()


# Drive forward, detect obstacles in front, try random directions to avoid
def autodrive_simple():
    dur = 10
    turnDur = 5

    # autodrive for 120 seconds
    for i in range(4):
        # forward
        frontCheck()
        init()
        mc.forward(dur)

        # left turn
        frontCheck()
        init()
        mc.turnLeft(turnDur)

        # forward
        frontCheck()
        init()
        mc.forward(dur)

        # right turn
        frontCheck()
        init()
        mc.turnRight(turnDur)


def main():
    init()
    autodrive_simple()


if __name__ == '__main__':
    main()
