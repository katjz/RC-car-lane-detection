import RPi.GPIO as gpio
import time

def frontDist(measure='cm'):
    gpio.setmode(gpio.BOARD)
    gpio.setup(12,gpio.OUT) #output pin
    gpio.setup(16,gpio.IN)  #input pin

    gpio.output(12,False) #reset pin 12

    # while pin 16 is False (no signal)...
    while gpio.input(16) == 0:
        noSignal = time.time()

    while gpio.input(16) == 1:
        signal = time.time()

    timeLength = signal - noSignal

    # calculate distance
    if measure == 'cm':
        dist = timeLength / 0.000058
    elif measure == 'in':
        dist = timeLength / 0.000148
    else:
        print("Invalid measurement unit")
        dist = None

    gpio.cleanup()
    return dist
