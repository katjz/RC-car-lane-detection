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

def mkCoords(img, line):
    #print(line)
    slope,intercept = line
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def avgSlopeInter(img,lines):
    left = []
    right = []
    for line in lines:
        #print(line)
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1,x2), (y1,y2), 1)
        slope = params[0]
        intercept = params[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
    leftAvg = np.average(left, axis=0)
    rightAvg = np.average(right, axis=0)
    #print(leftAvg)
    #print(rightAvg)
    leftLine = mkCoords(img, leftAvg)
    rightLine = mkCoords(img, rightAvg)
    return np.array([leftLine, rightLine])

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
    #plt.show()

def display2(img1, img2):
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()

def analyzeFrame(frame):
    img = frame
    cannyImg = canny(img)
    maskedImg = roi(cannyImg)
    lines = cv2.HoughLinesP(maskedImg,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    avgLines = avgSlopeInter(img, lines)
    lineImg = displayLines(img, avgLines)
    addImg = cv2.addWeighted(img, 0.8,lineImg,1,1)
    #display2(img, addImg)
    #display(addImg)
    cv2.imshow("result", addImg)
    cv2.waitKey(1)

def main():
    cap = cv2.VideoCapture("test2.mp4")
    while cap.isOpened():
        frame = cap.read()[1]
        analyzeFrame(frame)

main()
