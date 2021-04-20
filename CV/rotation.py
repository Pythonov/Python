import numpy as np
import argparse
import imutils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["input"])
cv2.imshow("Orig.", image)

(h, w) = image.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 45 deg.", rotated)

M = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow('Rotated by -90 deg.', rotated)

rotated = imutils.rotate(image, -30)
cv2.imshow("rotated automatically", rotated)
cv2.waitKey(0)
