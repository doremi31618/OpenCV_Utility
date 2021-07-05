import cv2
import numpy as np
import pathlib
from os.path import join

def main():
    file_path = pathlib.Path(__file__)
    for i in range(0,4):
        file_path = file_path.parent.resolve()
    
    img1_path = join(file_path, "image.jpg")
    img1 = cv2.imread(img1_path)

    rows, cols, channels = img1.shape
    '''affine transformation'''

    # translation & scaling
    # M = | s 0 tx |
    #     | 0 s ty |
    '''
    mat = np.float32([[1, 0, 100],[0, 1, 50]]) # translate matrix
    dst = cv2.warpAffine(img1, mat, (cols, rows)) # apply matrix
    '''
    
    # rotation 
    # M = | cos -sin |
    #     | sin  cos |
    angle = 90
    ''' Manually create matrix 
    row1 = np.float32([np.cos(angle * np.pi /180), -np.sin(angle  * np.pi /180), cols*3//4])
    row2 = np.float32([np.sin(angle * np.pi /180), np.cos(angle  * np.pi /180), 0])
    mat = np.block([[row1], [row2]])
    '''
    # the official version will auto correct the postion of image to assure the image is 
    # at center of screen
    mat = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), angle, 1)
    dst = cv2.warpAffine(img1, mat, (cols, rows))

    '''perspective transformation'''
    # step1 : find corner of transform target (these are 2d vertices)
    # step2 : find corner of destination image space
    # step3 : decide the size of image
    image_corners = np.float32()


    # display image
    cv2.imshow("img", img1)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # img1 = cv2.imread("")
    
if __name__ == '__main__':
    main()