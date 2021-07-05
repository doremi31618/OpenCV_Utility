import cv2 
import numpy as np

# from cv2 import FeatureDetector_create
cap = cv2.VideoCapture(1)
# detector =  cv2.FeatureDetector_create(""FAST")
while (True):  
    _, frame = cap.read()

    # find corner Harris 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    # gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray, 8, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    # frame[dst > 0.02*dst.max()]=[255, 0, 255]


    # find feature point 
    # sift = cv2.xfeatures2d.SIFT_create()
    # keypoints = sift.detect(gray, None)
    # cv2.imshow("",cv2.drawKeypoints(frame, keypoints, None, (255, 0, 255)))

    cv2.imshow("Streamming Live", frame)
    
    # print(kps)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()