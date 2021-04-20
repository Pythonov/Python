import cv2
import numpy as np

img = cv2.imread("Resources/dog.JPG")
print(img.shape)


def resizeImage(image, factor):
        imgResized = cv2.resize(image, (int((image.shape[1]*factor)),
                                        int((image.shape[0]*factor))))
        return imgResized


imgResize = resizeImage(img, 1)


grayImg = np.ndarray([imgResize.shape[0], imgResize.shape[1]], dtype="uint8")
grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayimage = resizeImage(grayimage, 1)


for h in np.arange(0, imgResize.shape[0]):
    for w in np.arange(0, imgResize.shape[1]):

        prom = np.sum(imgResize[h, w])/3
        grayImg[h, w] = prom.astype("uint8")
        #print(grayImg[h, w])


print(type(grayImg[100,100]))
print(type(grayimage[100,100]))
cv2.imshow("BW factory", grayimage)
cv2.imshow("BW", grayImg)
#cv2.imshow("Image", img)
#cv2.imshow("Image Resize", imgResize)

cv2.waitKey(0)

