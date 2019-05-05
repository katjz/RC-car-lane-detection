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

def roi(img):
    region = np.array([[(240, img.shape[0]),(1100, img.shape[0]),(550, 275)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,region,255)
    maskedImg = cv2.bitwise_and(img, mask)
    return maskedImg

def displayLines(img, lines):
    lineImg = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImg, (x1,y1),(x2,y2), (255, 0, 0), 10)
    return lineImg

# display image
def display(img):
    plt.imshow(img)
    plt.show()

def display2(img1, img2):
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()

def main():
    img = readImg()
    cannyImg = canny(img)
    maskedImg = roi(cannyImg)
    lines = cv2.HoughLinesP(maskedImg,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    lineImg = displayLines(img, lines)
    addImg = cv2.addWeighted(img, 0.8,lineImg,1,1)
    display2(img, addImg)

main()
