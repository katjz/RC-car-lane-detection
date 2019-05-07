import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip
from IPython.display import HTML

## Step 1: Canny edge detection
def canny(img):
    # step 3 pre-processing --> convert to grayscale
    grayImg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # complete step 3 --> canny edge detection
    img = cv2.Canny(img,100,200)
    return img

## Step #2: Restrict image to triangular ROI
def roi(img,vertices):
    # create blank matrix with same height/width
    mask = np.zeros_like(img)
    matchMaskColor = 255 # mask color for grayscale image
    # fill inside roi polygon
    cv2.fillPoly(mask,vertices,matchMaskColor)

    # return img only where mask pixels match
    maskedImg = cv2.bitwise_and(img,mask)
    return maskedImg

## Step #3: Hough Transform to detect calculate following edge detection
def hough(img):
    lines = cv2.HoughLinesP(
        img,
        rho=6,
        theta=np.pi/60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    #print(lines)
    #img = lineOverlay(img,lines)
    return lines

## Step 4: Extract single Left and Right lane lines
def groupLines(img,lines):
    minSlope = 0.5

    leftLine_x = []
    leftLine_y = []
    rightLine_x = []
    rightLine_y = []

    # create groupings of left and right lines; discard improbable lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1) / (x2-x1)   # calculate slope of line
            if math.fabs(slope) < minSlope:  # consider only near vertical slopes
                continue
            if slope <= 0:  # if slope is negative --> line is in left grouping
                leftLine_x.extend([x1,x2])
                leftLine_y.extend([y1,y2])
            else:           # if slope if positive --> line is in right grouping
                rightLine_x.extend([x1,x2])
                rightLine_y.extend([y1,y2])

    # Determine top and bottom points of each line
    min_y = img.shape[0] * (3/5) # min x --> just below horizon
    max_y = img.shape[0]        # max x --> bottom of image

    polyLeft = np.poly1d(np.polyfit(leftLine_y,leftLine_x,deg=1))
    left_x_start = int(polyLeft(max_y))
    left_x_end = int(polyLeft(min_y))

    polyRight = np.poly1d(np.polyfit(rightLine_y,rightLine_x,deg=1))
    right_x_start = int(polyRight(max_y))
    right_x_end = int(polyRight(min_y))

    leftLine = [left_x_start,max_y,left_x_end,int(min_y)]
    rightLine = [right_x_start,max_y,right_x_end,int(min_y)]

    return [leftLine,rightLine]


## Step 5: Overlays lines onto original image
def lineOverlay(img,lines,color=[0,255,0],thickness=4):
    # if there are no lines to draw, return
    if lines is None:
        return

    # Copy original image
    copy = np.copy(img)
    # Create blank image of same size
    lineImg = np.zeros((copy.shape[0],copy.shape[1],3), dtype=np.uint8)

    # Loop over lines and draw them on blank img
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(lineImg,(x1,y1),(x2,y2),color,thickness)

    # Merge line image with original img
    copy = cv2.addWeighted(copy,0.8,lineImg,1.0,0.0)

    return copy

# Helper function to display given image
def display(img):
    plt.imshow(img)
    plt.show()


### Pipeline Wrapper Function!
# 5-step pipeline for image processing to detect lane lines.
def pipeline(img):
    #wait = 1
    # complete step 1 (for still img) --> read in an image
    # img,height,width = readImg()

    # complete step 1 (for video) --> image passed in from video processor
    height = img.shape[0]
    width = img.shape[1]

    # complete step 2 --> canny edge detection
    cannyImg = canny(img)

    # define vertices for triangular ROI
    roi_vertices = [(0,height),(width/2,height/2),(width,height)]

    # complete step 3 --> crop image to ROI
    croppedImg = roi(cannyImg,np.array([roi_vertices],np.int32))

    # complete step 4 --> Hough transform
    lines = hough(croppedImg)

    # complete step 5 --> group and average lines to identify L and R lane lines
    laneLines = groupLines(img,lines)

    # complete step 6 --> overlay detected lane lines onto original image
    img = lineOverlay(img,[laneLines])
    #plt.figure()
    #display(img)
    #plt.imshow(img)
    #cv2.waitKey(wait)

    return img


## Use pipeline to process still image
def processImg():
    imgName = "./test_images/test_image.jpg"
    #imgName = "./test_images/solidWhiteCurve.jpg"
    img = cv2.imread(imgName)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pipeline(img,0) # wait is 0ms for still image

## Use pipeline to process video
def processVideo():
    waitTime = 1 # wait time of 1 ms for video read
    vid = cv2.VideoCapture("./test_video/test2.mp4") # load video
    while vid.isOpened():
        frame = vid.read()[1]
        pipeline(frame, waitTime)
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()

def saveVideo():
    white_output = 'solidWhiteRight_output.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(white_output, audio=False)

def main():
    #processImg()
    #processVideo()
    saveVideo()





if __name__ == '__main__':
    main()