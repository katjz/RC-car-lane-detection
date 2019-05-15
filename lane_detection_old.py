import cv2
import numpy as np
import matplotlib.pyplot as plt

## Initial version of lane detection
# not used in final version


def readImg():
    imgName = "test_image.jpg"#Loads test image
    img = cv2.imread(imgName)
    lane_img = np.copy(img)
    analyzeFrame(lane_img, 0)

def canny(img):#creates canny version of image, step 1 of pipeline
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)    # convert to grayscale
    blur = cv2.GaussianBlur(gray,(5,5),0)   # apply 5x5 gaussian
    canny = cv2.Canny(blur,50,150)
    return canny

def roi(img):#creates a mask of the region of interest over the initial image
    #step 2 of pipeline
    region = np.array([[(240, img.shape[0]),(1100, img.shape[0]),(550, 275)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,region,255)
    maskedImg = cv2.bitwise_and(img, mask)
    return maskedImg

def mkCoords(img, line): #creates the coordinates of a line for the image given a line
    slope,intercept = line
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])
        
    

def avgSlopeInter(img,lines):#finds the avg right and left line so we have a single lane line
    #step 4 of pipeline
    left = []#left lines array
    right = []#right lines array 
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
    ##handling for no left lines or no right lines
    if len(left)>0:
        leftAvg = np.average(left, axis=0)
        leftLine = mkCoords(img, leftAvg)
    else:
        leftLine = np.array([])
    if len(right)>0:
        rightAvg = np.average(right, axis=0)
        rightLine = mkCoords(img, rightAvg)
    else:
        rightLine = np.array([])
    if leftLine.shape[0]==0:
        if rightLine.shape[0]==0:
            return np.array([])
        else:
            return np.array([rightLine])
    else:
        if rightLine[0]==0:
            return np.array([leftLine])
        else:
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

def analyzeFrame(frame, wait):#arguments: frame is image, wait is wait time
    img = frame
    cannyImg = canny(img)#step 1
    maskedImg = roi(cannyImg)#step 2
    #step 3
    lines = cv2.HoughLinesP(maskedImg,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    avgLines = avgSlopeInter(img, lines)#step 4
    lineImg = displayLines(img, avgLines)
    addImg = cv2.addWeighted(img, 0.8,lineImg,1,1)
        #display2(img, addImg)
        #display(addImg)
    cv2.imshow("result", addImg)
    cv2.waitKey(wait)

def readVideo():
    waitTime = 1 # wait time of 1 ms for video read
    cap = cv2.VideoCapture("test2.mp4") #loads video
    while cap.isOpened():
        frame = cap.read()[1]
        analyzeFrame(frame, waitTime)
        if cv2.waitKey(1) == ord('q'):#quit out of program, only works sometimes
            break
    cap.release()
    cv2.destroyAllWindows()
        
def main():
    readVideo()

main()
