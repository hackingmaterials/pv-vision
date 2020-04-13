import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from lenet_model import LeNet


"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
"""

EPOCHS = 150
INIT_LR = 1e-3
BS = 128
CLASS_NUM = 2
norm_size = 64

label_dir = {
    "SN": 0,
    "PN": 0,
    "SC": 1,
    "PC": 1,
}

# Loading the data
print("[INFO] loading images...")
data = []
labels = []
imagePaths = sorted(list(paths.list_images("../Images/Training_Set")))  # load your image set
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath, 0)
    image = cv2.resize(image, (norm_size, norm_size))
    image = img_to_array(image)
    data.append(image)

    label_str = imagePath.split(os.path.sep)[-2]

    labels.append(label_dir[label_str])

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and validating splits using 75% of
# the data for training and the remaining 25% for validating
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25,
                                                  random_state=42)


# labels = to_categorical(labels, num_classes=CLASS_NUM)
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=CLASS_NUM)
testY = to_categorical(testY, num_classes=CLASS_NUM)

# construct the image generator for data augmentation
aug = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                         fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=norm_size, height=norm_size, depth=1,
                    classes=CLASS_NUM)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001) #automatically alter the learning rate
H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
              epochs=EPOCHS, verbose=1, callbacks=[reduce_lr])

# save the model to disk
print("[INFO] serializing network...")
model.save("crack.model")

# plot the loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Crack_noncrack(LeNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
