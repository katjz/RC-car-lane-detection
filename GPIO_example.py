import RPi.GPIO as gpio
import time

### Helpful Notes:
# 26 pins: top left = pin 1
# pins have names according to what they do
# BCM mode --> Broadcom GPIO numbers system. You can onle use one system per program
# Pins have HIGH/LOW values which correspond to on/off or 1/0


gpio.setmode(gpio.BCM)
gpio.setup(18,gpio.OUT)

while True:
    gpio.output(18,gpio.HIGH)
