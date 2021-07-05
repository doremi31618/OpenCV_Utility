''' description : Changing Colorspaces


'''
import cv2 as cv
import numpy as np

import os 
from os.path import join, isfile

def main():
    
        
    # get all the flags of color convention flag
    # flags = [i for i in dir(cv2) if i.startswith(('COLOR_'))]

    cap = cv.VideoCapture(1)
    while(True):
        # Take each frame
        _, frame = cap.read()
        # convert BGR to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # deform rang of blue color in HSV
        # create blue mask 
        lower_blue= np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv.inRange(hsv, lower_blue, upper_blue)

        # create red mask
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask_red = cv.inRange(hsv, lower_red, upper_red)
        
        # create green mask
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([70, 255, 255])
        mask_green = cv.inRange(hsv, lower_green, upper_green)

        # composite mask 
        mask = mask_blue | mask_red | mask_green
        # Biwise And mask and original image
        res = cv.bitwise_and(frame, frame, mask=mask)

        cv.imshow('frame', frame)
        cv.imshow('mask', mask)
        cv.imshow('res', res)
        k = cv.waitKey(5) & 0xFF
        if k==27:
            break
    cv.destroyAllWindows()

if __name__ == '__main__': 
    main()