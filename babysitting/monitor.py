import matplotlib
matplotlib.use("Agg")

from babysitting.callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from convolutional import miniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output dir.")
args = vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid()))
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = miniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

figPath = os.path.sep.join([args["output"], "{}.png".format(
    os.getpid())])

jsonPath = os.path.sep.join([args["output"], "{}.json".format(
    os.getpid())])

callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY),
          batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
