import RPi.GPIO as gpio
import test_image

# PIN NUMBER CONSTANTS
# Wheels denoted as: F = front, B = back, L = left, R = right
FL = 7  # input 4
FR = 11 # input 3
BL = 13 # input 2
BR = 15 # input 1


# Init Function: sets up the pins we will be using
# it's best to run init() inside every function to reset all pins
def init():
    gpio.setmode(gpio.BOARD)
    gpio.setup(7,gpio.OUT)
    gpio.setup(11, gpio.OUT)
    gpio.setup(13, gpio.OUT)
    gpio.setup(15, gpio.OUT)
    #gpio.output(7, True)
    #gpio.output(11, True)

# Move car forward for dur seconds
def forward(dur):
    init()
    gpio.output(7,False)
    gpio.output(11,True)
    gpio.output(13,True)
    gpio.output(15,False)
    time.sleep(dur)
    gpio.cleanup()

# Move car backwards for dur seconds
def reverse(dur):
    init()
    gpio.output(7,True)
    gpio.output(11,False)
    gpio.output(13,False)
    gpio.output(15,True)
    time.sleep(dur)
    gpio.cleanup()

# Turn car left for dur seconds
def turnLeft(dur):
    init()
    gpio.output(7,False)
    gpio.output(13,False)
    gpio.output(15,False)
    gpio.output(11,True)
    time.sleep(dur)
    gpio.cleanup()

# Turn car right for dur seconds
def turnRight(dur):
    init()
    gpio.output(7,False)
    gpio.output(13,False)
    gpio.output(15,False)
    gpio.output(11,True)
    time.sleep(dur)
    gpio.cleanup()

# Pivot left for dur seconds (pivot gains no forward ground)
def pivotLeft(dur):
    init()
    gpio.output(7,True)
    gpio.output(11,False)
    gpio.output(13,True)
    gpio.output(15,False)
    time.sleep(dur)
    gpio.cleanup()

# Pivot right for dur seconds (pivot gains no forward ground)
def pivotReft(dur):
    init()
    gpio.output(7,False)
    gpio.output(11,True)
    gpio.output(13,False)
    gpio.output(15,True)
    time.sleep(dur)
    gpio.cleanup()

def main():
    dur = 1
    init()
    forward(dur)
    reverse(dur)
    pivotLeft(dur)
    pivorRight(dur)
    gpio.cleanup() # cease pins from being activated

if __name__ == '__main__':
    main()
