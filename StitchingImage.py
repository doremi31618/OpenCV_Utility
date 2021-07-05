import cv2 

from os import listdir
from os.path import isfile, join


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

path = "images/"
onlyImages = [cv2.imread(path + f) for f in listdir(path) if isfile(join(path,f))]
# print(onlyImages)
stitcher = cv2.Stitcher.create()
status, result = stitcher.stitch(onlyImages)
(h, w) = result.shape[:2]
img = image_resize(result, width = 1920)

while(True):
    cv2.imshow("stitcher", img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    