from __future__ import print_function
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["input"])
cv2.imshow("Orig.", image)

(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r,
                                                              g, b))

image[0, 0] = (0, 255, 0)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r,
                                                              g, b))

corner = image[0:100, 0:100]
cv2.imshow("Corner", corner)

image[0:100, 0:100] = (0, 255, 0)

cv2.imshow("Updated", image)
cv2.waitKey(5000)
