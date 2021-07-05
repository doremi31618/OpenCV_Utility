import cv2
import numpy as np
import pathlib
from os.path import join

def main():
    file_path = pathlib.Path(__file__)
    for i in range(0, 4):
        file_path = file_path.parent.resolve()
    
    img1_path = join(file_path, "image.jpg")
    img1 = cv2.imread(img1_path)
    

if __name__ == '__main__':
    main()