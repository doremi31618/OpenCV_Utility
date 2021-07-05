import cv2
import os
from pathlib import Path
from os.path import join
def main():
    script_path = os.getcwd()
    path = 'C:\\Users\\panos\\Desktop\\PanoAWS\\Projects\\Opencv_algorithm\\result\\result.jpg_screenshot_28.06.2021.png'
    print(path)
    img_origin = cv2.imread(path)
    cv2.imshow("origin", img_origin)
    img_resize = cv2.resize(img_origin, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR_EXACT)
    cv2.imshow("resize", img_resize)
    cv2.waitKey()

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()