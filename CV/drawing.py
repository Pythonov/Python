import numpy as np
import cv2


canvas = np.zeros((300, 300, 3), dtype="uint8")
cv2.imshow("Canvas", canvas)

green = (0, 255, 0)
cv2.line(canvas, (0, 0), (300, 300), green)
cv2.imshow("Green", canvas)


red = (0, 0, 255)
cv2.line(canvas, (300, 0), (0, 300), red, 3)
cv2.imshow("Red", canvas)


cv2.rectangle(canvas, (10, 10), (60, 60), red)
cv2.imshow("Canvas2", canvas)


canvas = np.zeros((300, 300, 3), dtype="uint8")
(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
color = (200, 100, 50)
for r in range(0, 175, 10):
    cv2.circle(canvas, (centerX, centerY), r, color)

cv2.imshow("circles", canvas)

for i in range(0, 25):
    radius = np.random.randint(5, high=200)
    color = np.random.randint(0, high=256, size=(3,)).tolist()
    pt = np.random.randint(0, high=300, size=(2,))

    cv2.circle(canvas, tuple(pt), radius, color, -1)

cv2.imshow("canvas", canvas)
cv2.waitKey(0)


