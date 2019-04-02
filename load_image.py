import cv2
import numpy as np
import matplotlib.pyplot as plt

def readImg():
    imgName = "test_image.jpg"
    img = cv2.imread(imgName)
    lane_img = np.copy(img)
    return lane_img

def canny(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)    # convert to grayscale
    blur = cv2.GaussianBlur(gray,(5,5),0)   # apply 5x5 gaussian
    canny = cv2.Canny(blur,50,150)
    return canny

#def roi(img):
#    triangle = np.array([])

# display image
def display(img):
    plt.imshow(img)
    plt.show()


def main():
    img = readImg()
    cannyImg = canny(img)
    display(cannyImg)

main()


