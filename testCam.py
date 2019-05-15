from picamera import PiCamera
from time import sleep

##tests input from camera, just streams video to screen and stops after a while 

camera = PiCamera()

camera.start_preview()
sleep(30)
camera.stop_preview()
