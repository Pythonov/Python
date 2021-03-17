
import cv2

image = cv2.imread("C:\code\ORMAL2-IM-0372-0001.jpeg")
print(image.shape)

print('\n============================\n')
# openCV stores data in reversed format!!!(bgr)

(b,g,r) = image[20,100]  # y = 20,x = 100
(b,g,r) = image[750,1500]
(b,g,r) = image[1000,1000]

cv2.imshow("Zennenhund", image)
cv2.waitKey(0)
