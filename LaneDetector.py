import sys
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
from IPython.display import HTML

###################### FINAL VERISON --> TEST THIS ONE! :) #####################

###############################################################################
# Encapsulated functions for each step in the lane detection Pipeline
###############################################################################

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
    return lines

## Step 4: Extract single Left and Right lane lines
def groupLines(img,lines):
    # if there are no lines to draw, return
    if lines is None:
        return img

    # Ignore lines with slope < 0.5. We are only concerned with lines that are closer to vertical.
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

    # for lines in L and R groupings, respectively, calculate averaged lane line
    try:
        polyLeft = np.poly1d(np.polyfit(leftLine_y,leftLine_x,deg=1))
        left_x_start = int(polyLeft(max_y))
        left_x_end = int(polyLeft(min_y))


        polyRight = np.poly1d(np.polyfit(rightLine_y,rightLine_x,deg=1))
        right_x_start = int(polyRight(max_y))
        right_x_end = int(polyRight(min_y))

    # If we are unable to perform averaging calculations due to lack of probable lines
    # in either grouping, return None for L and R lane line
    except TypeError:
        return None

    leftLine = [left_x_start,max_y,left_x_end,int(min_y)]
    rightLine = [right_x_start,max_y,right_x_end,int(min_y)]

    return [leftLine,rightLine]


## Step 5: Overlays lines onto original image
def lineOverlay(img,lines,color=[0,255,0],thickness=4):
    # if there are no lines to draw, return
    if lines is None:
        return None
    for line in lines:
        if line is None:
            return None

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

    # Create blank image of same size
    fillImg = np.zeros((copy.shape[0],copy.shape[1],3), dtype=np.uint8)
    # fill in region between lane lines
    pairs=[[]]
    for line in lines[0]:
        for coord in line:
            pairs[0].append(coord)
    for x1,y1,x2,y2,x3,y3,x4,y4 in pairs:
        pts = np.array([[x1,y1],[x2,y2],[x4,y4],[x3,y3]], np.int32)
        cv2.fillConvexPoly(fillImg, pts, [0,255,0,50])

    # Merge line image with original img
    copy = cv2.addWeighted(fillImg,0.3,copy,1.0,0.0)

    return copy
###############################################################################
# End of pipeline functions
###############################################################################

###############################################################################
# Functions for determining/reporting lane and camera centers
###############################################################################
##
def getCenter(camCenter_x,laneLines):
    l_line = laneLines[0]
    r_line = laneLines[1]
    # get x coordinate of start endpoint of L and R lane lines
    xL = l_line[0]
    xR = r_line[0]
    laneCenter_x = xL + (xR-xL)/2
    #print("Center: ", laneCenter_x)

    # compute number of pixels off center
    offCenter = laneCenter_x - camCenter_x

    # get x coordinate of horizon endpoint of L and R lane laneLines
    xL = l_line[2]
    xR = r_line[2]
    topCenter_x = xL + (xR-xL)//2

    # build center line coordinates
    centerLine = [int(laneCenter_x),l_line[1],topCenter_x,l_line[3]]
    return offCenter,centerLine

def drawCenter(img,camCenter,laneCenter,offCenter):
    # Copy original image
    #copy = np.copy(img)
    # Create blank image of same size
    #lineImg = np.zeros((copy.shape[0],copy.shape[1],3), dtype=np.uint8)

    pixels = int(abs(offCenter))
    text = ""
    if offCenter > 0:
         text = "Car is " + str(pixels) + " pixels left of center."
    elif offCenter < 0:
         text = "Car is " + str(pixels) + " pixels right of center."

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thick = 3
    cv2.putText(img, text, (50,50), font, scale, (255,255,255),thick, cv2.LINE_AA)

    # Loop over lines and draw them on blank img
    for x1,y1,x2,y2 in laneCenter:
        cv2.line(img,(x1,y1),(x2,y2),[255,255,0],4)
    for x1,y1,x2,y2 in camCenter:
        cv2.arrowedLine(img,(x1,y1),(x2,y2),[255,0,0],4)

    # Merge line image with original img
    #copy = cv2.addWeighted(copy,0.8,lineImg,1.0,0.0)

    return img
###############################################################################
# End center calculation functions
###############################################################################


######################### Pipeline Wrapper Function! ##########################
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
    overlayImg = lineOverlay(img,[laneLines],)
    # if we do have L and R lines to work with
    if overlayImg is not None:
        # get coordinate of center of image
        center_x = width//2
        offCenter,laneCenterLine = getCenter(center_x,laneLines)
        camCenterLine = [center_x,height,center_x,int(3*height/4)]
        overlayImg = drawCenter(overlayImg,[camCenterLine],[laneCenterLine],offCenter)
    else:
        overlayImg = img

    #display(overlayImg)

    return overlayImg
###############################################################################

# Helper function to display given image
def display(img):
    plt.imshow(img)
    plt.show()

###############################################################################
# Wrapper functions to run pipeline with still image & video
###############################################################################
## Use pipeline to process still image
def processImg(fName):
    img = cv2.imread(fName)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = pipeline(img)

    display(img)

## Use pipeline to process video and save processed video to folder
def saveVideo(fName):
    output_name = "./video_output/" + fName[13:]
    input_vid = VideoFileClip(fName)
    output_vid = input_vid.fl_image(pipeline)
    output_vid.write_videofile(output_name, audio=False)

def main():
    if len(sys.argv) < 2:
        done = False
        while not done:
            print()
            print("Input the name of the file to be processed or the number corresponding to one of the suggested files.")
            print("Videos:")
            print("1) ./test_video/solidYellowLeft.mp4")
            print("2) ./test_video/solidWhiteRight.mp4")
            print("3) ./test_video/challenge.mp4")
            print("Images:")
            print("4) ./test_images/solidYellowCurve.jpg")
            print("5) ./test_images/solidWhiteCurve.jpg")
            print("6) ./test_images/test_image.jpg")

            fName = input("File Name: ")
            if fName == "1":
                fName = "./test_video/solidYellowLeft.mp4"
            elif fName == "2":
                fName = "./test_video/solidWhiteRight.mp4"
            elif fName == "3":
                fName = "./test_video/challenge.mp4"
            elif fName == "4":
                fName = "./test_images/solidYellowCurve.jpg"
            elif fName == "5":
                fName = "./test_images/solidWhiteCurve.jpg"
            elif fName == "6":
                fName = "./test_images/test_image.jpg"

            exists = os.path.isfile(fName)
            if exists:
                done = True
            else:
                print("\nFile not found.\n")

    else:
        fName = sys.argv[1]

        exists = os.path.isfile(fName)
        if exists:
            done = True
        else:
            print("\nFile not found.\n")

    # check file type and call correct wrapper function
    if fName[-3:] == "jpg":
        processImg(fName)
    elif fName[-3:] == "mp4":
        saveVideo(fName)
    else:
        print("Error: input file of type .jpg or .mp4")


if __name__ == '__main__':
    main()
