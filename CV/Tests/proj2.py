import cv2
import numpy as np


img = cv2.imread("Resources/dog.JPG")
kernel = np.ones((3, 3), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgCanny = cv2.Canny(img, 100, 100)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

cv2.imshow("Original", img)
#cv2.imshow("DogGray", imgGray)
#cv2.imshow("Blur image", imgBlur)
cv2.imshow("Canny image", imgCanny)
cv2.imshow("Dilation image", imgDialation)
cv2.imshow("Eroded image", imgEroded)
cv2.waitKey(0)
