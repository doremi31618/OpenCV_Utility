import numpy as np
import cv2

### This Example only show two image stick together 

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

if __name__ == '__main__':
    # Read Images
    img1 = cv2.imread ("./Images/ICL_8196.JPG")
    img2 = cv2.imread ("./Images/ICL_8206.JPG")
    
    # img1 = cv2.imread ("./Sticher_test/2O1A9185.jpg")
    # img2 = cv2.imread ("./Sticher_test/2O1A9188.jpg")

    # convert rgb img to gray scale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Affine img1 
    xoffset, yoffset= 100, 50
    (h, w) = gray_img1.shape

    # (SIFT) find keypoints and description 
    sift = cv2.xfeatures2d.SIFT_create() # the coeficient in the function is Hessian Threshold

    kp1, des1 = sift.detectAndCompute(gray_img1,None)
    des_img1 = cv2.drawKeypoints(gray_img1, kp1, None, (255, 0, 0), 4)

    kp2, des2 = sift.detectAndCompute(gray_img2,None)
    des_img2 = cv2.drawKeypoints(gray_img2, kp2, None, (255, 0, 0), 4)

    # match keypoint
    # FLANN_INDEX_KDTREE = 1
    # index_parms = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_parms = dict(check = 50)

    # flann = cv2.FlannBasedMatcher(index_parms, search_parms)
    # matches = flann.knnMatch(des1, des2, k=2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Store good matches 
    good_match = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_match.append(m)

    # draw match 
    # draw_parms = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = None, flags = 2)
    # result = cv2.drawMatches (des_img1, kp1, des_img2, kp2, good_match, None, **draw_parms)

    # find homograhy matrix 
    img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good_match ]).reshape(-1, 1, 2)
    img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good_match ]).reshape(-1, 1, 2)
    
    homograhy_mat, mask = cv2.findHomography(img2_pts, img1_pts, cv2.RANSAC)
    result = cv2.warpPerspective(img2, homograhy_mat, (w + gray_img2.shape[0],h))
    result[0:img1.shape[0], 0:img1.shape[1]] = img1

    # Resize Image 
    result = image_resize(result, width=1920)

    # show result
    while(True):
        cv2.imshow("SURF matching - result", result)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break