#import matplotlib
#matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from convolutional import miniVGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# =============== args parsing ====================
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
ap.add_argument("-d", "--dataset", required=True,
                help="path to the dataset dir.")
ap.add_argument("-m", "--model", required=True,
                help="path to the output model")
args = vars(ap.parse_args())
# =============== args parsing ====================

print("[INFO] loading dataset...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=250)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25,
                                                  random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=1e-4)
model = miniVGGNet.build(32, 32, 3, 3)
model.compile(optimizer=opt, loss="categorical_crossentropy",
              metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit(trainX, trainY, batch_size=64, epochs=100,
              verbose=1, validation_data=(testX, testY))

print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=["cats", "dogs", "pandas"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
