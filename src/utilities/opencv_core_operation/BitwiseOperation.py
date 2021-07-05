import cv2 

def main():
    # Use case :
    # Assuem you want to put your logo onto the target image 
    # ( for convinience we name target image to main image )

    # Bitwise operation work flow
    # 1. Read Image 
    # 2. Create ROI (range of interest) of logo image
    # 3. Create a mask for logo
    # 4. Perform bitwise operator
    # 5. replace roi region to main image

    # read image
    img1 = cv2.imread('C:\\Users\panos\Desktop\PanoAWS\Projects\Opencv_algorithm\All_image\ICL_8191.JPG')
    img2 = cv2.imread('C:\\Users\panos\Desktop\PanoAWS\Projects\Opencv_algorithm\All_image\logo.png')

    # create roi
    row, col, channel = img2.shape
    roi = img1[0:row, 0:col]

    # create mask
    ### convert color to gray scale ( mask 只能是單通道 )
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)#### 運作原理 : 如果大於thresh，就賦予255的值
    mask_inv = cv2.bitwise_not(mask)#### 對mask做not運算a -> 255-a ; 255-a -> a

    # perform bitwise operation
    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask = mask)

    result = cv2.add(img1_bg, img2_fg)
    img1[0:row, 0:col] = result
    cv2.imshow("img1 result",img1)
    cv2.waitKey()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()