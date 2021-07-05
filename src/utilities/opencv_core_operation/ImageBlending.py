import cv2
import os
from os.path import join, isfile

def main():
    img1 = cv2.imread('C:\\Users\panos\Desktop\PanoAWS\Projects\Opencv_algorithm\All_image\ICL_8251.JPG')
    img2 = cv2.imread('C:\\Users\panos\Desktop\PanoAWS\Projects\Opencv_algorithm\All_image\ICL_8226.JPG')

    # Image blending
    dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()