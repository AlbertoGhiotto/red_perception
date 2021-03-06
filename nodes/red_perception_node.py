#! /usr/bin/env python

# Python libs
import sys, time

import time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
from cv2 import cv2

# Image processing
import matplotlib.pyplot as plt
from skimage import io, filters, measure, color, external

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

#  To convert ros images to cv images 
from cv_bridge import CvBridge

# For arctan computation
import math


# _____________________________________________________________________

# Global variables
VERBOSE=True
DISPLAY=True

# _____________________________________________________________________
# "Callback class"

class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        #self.image_pub = rospy.Publisher("/output/image_raw/compressed",
        #    CompressedImage, queue_size=1)
        self.redCoord_pub = rospy.Publisher("redCoord", Float32MultiArray, queue_size=1)
        # self.redY_pub = rospy.Publisher("redY", Float32, queue_size=1)
        
        # init the ros bridge
        self.bridge = CvBridge()
        # subscribed Topic
        self.subscriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.callback,  queue_size=1)
        if VERBOSE :
            print("subscribed to /camera/color/image_raw")

        # define color ranges in HSV
        self.lower_red = np.array([0, 120, 50])
        self.upper_red = np.array([10, 255, 255])
        # self.lower_blue = np.array([110, 50, 50])
        # self.upper_blue = np.array([130, 255, 255])
        # self.lower_green = np.array([45, 140, 50])
        # self.upper_green = np.array([75, 255, 255])

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        # if VERBOSE :
        #     # print('received image of type: "%s"' % ros_data.encoding)
        
        time.sleep(0.25)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # img_str = cv2.imencode('.jpg', ros_data.data)[1] #Encodes and stores in buffer
        # print(img_str)
        #  np_arr = np.frombuffer(img_str, np.uint8)

        # np_arr = np.fromstring(cv_image, np.uint8)
       
        # print(np_arr)
        # Direct conversion to CV2
        # print("Cv2 ver: " + cv2.__version__)

        # frame = cv2.imdecode(cv_image, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        
        frame = cv_image

        # print(frame)
        # Get frame dimensions
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        frame_channels = frame.shape[2]

        #  Alternatively, the image msg already has these fields
        # if VERBOSE:
        #     # print("\nCV2 decoded data is:\n H: " + str(frame_height) + "\tW: " + str(frame_width))
        #     print("\nImage msg data is:  \nH: " + str(ros_data.height) + "\tW: " + str(ros_data.width))

        # Image processing
        res = self.red_filtering(frame)
        # Detect centrode
        (cX, cY, fullArea) = self.detect_centrode(res)
        # Write the point (cX,xY) on "res" image
        try:
            cv2.circle(res, (int(cX),int(cY)), 5, (255, 0, 0), -1)
        except ValueError as error :
            # E.g. cannot convert float NaN to integer
            pass
        # Normalizing w.r.t the center
        # cX = int(cX-frame_width/2) 
        # cY = int(cY-frame_height/2)

        # flag signaling if centroid is found
        found = False 

        # if ((cX != -frame_width/2) and (cY != -frame_height/2)): # centroid is found
        if ((cX != 0) and (cY != 0)): # centroid is found
            found = True

        center = (frame_width / 2  , frame_height /2 )

        cmd = self.decideCommand(cX, cY, fullArea, center)

        redData = (cmd, fullArea, found)    # publish command, red area and whether centrode is found

        msg = Float32MultiArray()
        msg.data = redData

        self.redCoord_pub.publish(msg)
        print("\nPublished red data on /redCoord!\n")

        # Print the center of mass coordinates w.r.t the center of image and display it
        if VERBOSE:
            if (found):
                print("Red centrode is: (" + str(cX) + "," + str(cY) + ")")
                print("Red area is: " + str(fullArea))
                print("Command: " + str(cmd))
            else:
                print("Red is not found!\n\n")
            print("__________")
        # Display the result
        if DISPLAY:
            imS = cv2.resize(frame, (240, 240))                    # Resize image for displaying
            cv2.imshow("output", imS)                              # Show image
            imS = cv2.resize(res, (240, 240))                    # Resize image for displaying
            cv2.imshow("output", imS)                              # Show image
            # cv2.imshow('frame', frame)
            # cv2.imshow("res_center",res)
            cv2.waitKey(2)


    def red_filtering(self,frame):        
        # Adapted from 
        # https://stackoverflow.com/questions/54425093/
        # /how-can-i-find-the-center-of-the-pattern-and-the-distribution-of-a-color-around)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Threshold the HSV image to get only desired colors
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # Smooth and blur "res" image
        res = cv2.medianBlur(res,15)
            # Or also:
            # res = cv2.GaussianBlur(res,(15,15),0)
            # res = cv2.bilateralFilter(res,15,75,75)
        # Erode and convert it for noise reduction
        kernel = np.ones((2,2),np.uint8)
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        res = cv2.convertScaleAbs(res)
        return res

    def detect_centrode(self,res):        
        # Adapted from 
        # https://stackoverflow.com/questions/54425093/
        # /how-can-i-find-the-center-of-the-pattern-and-the-distribution-of-a-color-around)
        im2, contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        centersX = []
        centersY = []
        for cnt in contours:
            areas.append(cv2.contourArea(cnt))
            M = cv2.moments(cnt)
            try:
                centersX.append(int(M["m10"] / M["m00"]))
                centersY.append(int(M["m01"] / M["m00"]))    
            except ZeroDivisionError as error:
                # Output expected ZeroDivisionErrors.
                centersX.append(int(M["m10"]))
                centersY.append(int(M["m01"]))   
                pass    
        full_areas = np.sum(areas)
        acc_X = 0
        acc_Y = 0
        for i in range(len(areas)):
            acc_X += centersX[i] * (areas[i]/full_areas) 
            acc_Y += centersY[i] * (areas[i]/full_areas)

        return acc_X,acc_Y, full_areas

    
    def decideCommand(self, cX, cY, area, center):
        # dictionary :
        # 0:A ok!Go Ahead
        # 1:R centrode is on right
        # 2:L centrode is on left
        # 3:U centrode is up  
        # 4:D centrode is down
        # 5:ERR error
        
        m = (cY - center[1]) / (cX - center[0])   # angular coefficient of line from origin to centrode
        # print("M is: " + str(m))
        angle =  math.atan(m)        
        print("Angle is: " + str(round(angle,2)) + " RAD | " + str(np.rad2deg(round(angle,2))) + " GRAD")

        if( 260 < cX < 380 and 180 < cY < 300 ): # centrode is centered
            return 0
        elif( cX < center[0] ):    # Left side
            if(-math.pi / 4 < angle <  math.pi / 4):
                return 2 #LEFT
            elif(angle < - math.pi / 4):
                return 4 #DOWN
            elif(angle > math.pi / 4):
                return 3 #UP
        elif( cX >= center[0]):    # Right side
            if(-math.pi / 4 < angle <  math.pi / 4):
                return 1 #RIGHT
            elif(angle < - math.pi / 4):
                return 3 #UP    Inverted wrt left side
            elif(angle > math.pi / 4):  
                return 4 #DOWN
        else:
            return 5



# _____________________________________________________________________


def main(args):
    '''Initializes and cleanup ros node'''
    print("Starting ROS Image feature detector module")
    rospy.init_node('red_perception_node')
    ic = image_feature()
    r = rospy.Rate(1) # 1hz
    try:
        rospy.spin()
        r.sleep()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main(sys.argv)
