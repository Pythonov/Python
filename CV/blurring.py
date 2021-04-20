import numpy as np
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

resized = cv2.resize(image, (300,
                             int(300 * image.shape[0] / image.shape[1])))
cv2.imshow("Resized", resized)

# Average blurring
blurred = np.hstack([
    cv2.blur(resized, (3, 3)),
    cv2.blur(resized, (5, 5)),
    cv2.blur(resized, (7, 7))])
cv2.imshow("Averaged", blurred)


# Gaussian blurring
blurred = np.hstack([
    cv2.GaussianBlur(resized, (3, 3), 0),
    cv2.GaussianBlur(resized, (5, 5), 0),
    cv2.GaussianBlur(resized, (7, 7), 0)])
cv2.imshow("Gaussian", blurred)


# Median blurring
blurred = np.hstack([
    cv2.medianBlur(resized, 3),
    cv2.medianBlur(resized, 5),
    cv2.medianBlur(resized, 7)])
cv2.imshow("Median", blurred)


# Bilateral blurring
blurred = np.hstack([
    cv2.bilateralFilter(resized, 5, 21, 21),
    cv2.bilateralFilter(resized, 7, 31, 31),
    cv2.bilateralFilter(resized, 9, 41, 41)])
cv2.imshow("Bilateral", blurred)
cv2.waitKey(0)
