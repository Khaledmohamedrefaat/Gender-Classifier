# REQUIREMENTS
# pip install numpy opencv-python requests progressbar pillow tensorflow theano keras
# XML Files -> haarcascade_frontalface_default.xml - haarcascade_frontalface_alt.xml - haarcascade_frontalface_alt2.xml - haarcascade_frontalface_alt_tree.xml
# Pre-Trained Model -> preTrainedModel.h5


# Step 1 - Downloading the Dataset
# Dataset 1 Link (Separated): https://s3.ap-south-1.amazonaws.com/arunponnusamy/pre-trained-weights/gender_detection.model
# Dataset 2 Link (not Separated): https://drive.google.com/drive/folders/0BxYys69jI14kSVdWWllDMWhnN2c
---------------------------------------------------------------------------------------------------------------------------------------
# Step 2 - Separating Dataset 2
import os
import shutil

data = []

cntrmale = 0
cntrfemale = 0

fileList = os.listdir('part1')
for filename in fileList:
    if filename.endswith('.jpg'):
        filenameList = filename.split('_')
        if filenameList[1] == '0':
            if cntrmale % 10 == 8 or cntrmale % 10 == 9:
                shutil.move('part1/' + filename, 'part1/test_set/male/' + filename)
            else:
                shutil.move('part1/' + filename, 'part1/training_set/male/' + filename)
            cntrmale += 1
        else:
            if cntrfemale % 10 == 8 or cntrfemale % 10 == 9:
                shutil.move('part1/' + filename, 'part1/test_set/female/' + filename)
            else:
                shutil.move('part1/' + filename, 'part1/training_set/female/' + filename)
            cntrfemale += 1
---------------------------------------------------------------------------------------------------------------------------------------
# Step 3 - Cropping Faces from the Dataset
import os
import numpy as np
import cv2 as cv
cntr = 1
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')

allFiles = os.listdir('.')

for filename in allFiles:
    if filename.endswith('.jpg'):
        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            extractedImage = np.copy(img[y:y+h, x:x+w])
            name = 'image' + str(cntr) + '.jpg'
            print(name)
            cv.imwrite(name, extractedImage)
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = img[y : y + h, x : x + w]
            cntr += 1
            print(cntr)
---------------------------------------------------------------------------------------------------------------------------------------
# Step 4 - Creating and training Model
# Training Imports
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import random
import cv2
import os
import glob

# Global Variables
lerningRate = 1e-3
numEpochs = random.randint(100, 200)
currOptimizer = Adam(lr = lerningRate, decay = lerningRate/numEpochs)
datasetLocation = "newDataset/**/*"
image_dimensions = (96, 96, 3) # (height, width, depth)
numBatches = 32
Xmatrix = []
Yvector = []

# Part 1 - Filling the dataset with the files
# Adding Paths to Filelist
filesList = [filename for filename in glob.glob(datasetLocation, recursive = True) if not os.path.isdir(filename)]
random.seed(random.randint(40, 80))
random.shuffle(filesList)

# Resizing Each Photo and appending it to dataset
cntr = 0
for filename in filesList:
    image = cv2.imread(filename)
    image = cv2.resize(image, (image_dimensions[0], image_dimensions[1]))
    image = img_to_array(image)
    image = np.array(image, dtype = "float") / 255.0
    Xmatrix.append(image)
    currlabel = filename.split(os.path.sep)[-2]
    currlabel = 0 if currlabel == "man" else 1
    Yvector.append([currlabel])
    cntr += 1
    print(cntr)

# Part 2 - Data Pre-processing
Xmatrix = np.array(Xmatrix, dtype = "float")
Yvector = np.array(Yvector)
(X_train, X_test, Y_train, Y_test) = train_test_split(Xmatrix, Yvector, test_size = 0.2,random_state = random.randint(40, 80))
Y_train = to_categorical(Y_train, num_classes = 2)
Y_test = to_categorical(Y_test, num_classes = 2)

# Part 3 - Dataset Augmentation
data_augemntation = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Part 4 - Creating Model
# Model Imports
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

# Model Creation
# Initializing Model
model = Sequential()

# Layer 1 - Convolution-Normalizing-MaxPooling - Dropout 0.25
model.add(Conv2D(32, (3,3), padding = "same", input_shape = (96, 96, 3), activation = 'relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

# Layer 2 - Convolution-Convolution-Normalizing-MaxPooling - Dropout 0.25
model.add(Conv2D(64, (3,3), padding = "same", activation = 'relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3,3), padding = "same", activation = 'relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Layer 3 - Convolution-Convolution-Normalizing-MaxPooling - Dropout 0.25
model.add(Conv2D(128, (3,3), padding = "same", activation = 'relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3,3), padding = "same", activation = 'relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Layer 4 - Convolution-Convolution-Normalizing-MaxPooling - Dropout 0.5
model.add(Conv2D(256, (3,3), padding = "valid", activation = 'relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(256, (3,3), padding = "valid", activation = 'relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# Layer 5 - Flattening
model.add(Flatten())

# Layer 6 - FCC Input Layer - Dropout 0.5
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Layer 7 - FCC Output Layer - Activation Softmax
model.add(Dense(2))
model.add(Activation("softmax"))

# Compiling
model.compile(optimizer = currOptimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

# Model Training
trainedModel = model.fit_generator(data_augemntation.flow(X_train, Y_train, batch_size = numBatches),
                        validation_data = (X_test, Y_test),
                        steps_per_epoch = len(X_train) // numBatches,
                        epochs = numEpochs, verbose = 1)

# Saving Model - Weights
model.save('PreTrainedModel.h5')
# model.save_weights('savedModel.h5')
---------------------------------------------------------------------------------------------------------------------------------------
# Step 5 - Detect Input Camera Frames
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import os
import numpy as np
import cv2 as cv

model = load_model('preTrainedModel.h5')
genderWindow = "Gender Classification"
cv.namedWindow(genderWindow, cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty(genderWindow, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
webcam = cv.VideoCapture(0)
while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        print("Could not read frame")
        exit()
    img = frame
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        rectangleColor = (0, 255, 0)
        cv.rectangle(img, (x, y), (x + w, y + h), rectangleColor, 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]
        extractedImage = np.copy(img[y : y + h, x : x + w])
        if (extractedImage.shape[0]) < 10 or (extractedImage.shape[1]) < 10:
            continue
        extractedImage = cv.resize(extractedImage, (96, 96))
        extractedImage = extractedImage.astype("float") / 255.0
        extractedImage = img_to_array(extractedImage)
        extractedImage = np.expand_dims(extractedImage, axis = 0)
        pred = model.predict(extractedImage)
        if pred[0][0] > 0.5:
            label = "Male"
            textColor = (255, 0, 0)
        else:
            label = "Female"
            textColor = (0, 0, 255)
        cv.putText(img, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, Color, 2)

    cv.imshow(window_name, img)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv.destroyAllWindows()
---------------------------------------------------------------------------------------------------------------------------------------
# Step 6 - Detect An input image
# Output image is at -> output.jpg
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2 as cv
import os
import numpy as np
import cv2 as cv

model = load_model('preTrainedModel.h5')
path = ""
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
img = cv.imread(path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    rectangleColor = (255, 0, 0)
    cv.rectangle(img, (x, y), (x + w, y + h), rectangleColor, 2)
    roi_gray = gray[y : y + h, x : x + w]
    roi_color = img[y : y + h, x : x + w]
    extractedImage = np.copy(img[y : y + h, x : x + w])
    if (extractedImage.shape[0]) < 10 or (extractedImage.shape[1]) < 10:
        continue
    extractedImage = cv.resize(extractedImage, (96, 96))
    extractedImage = extractedImage.astype("float") / 255.0
    extractedImage = img_to_array(extractedImage)
    extractedImage = np.expand_dims(extractedImage, axis = 0)
    pred = model.predict(extractedImage)
    if pred[0][0] > 0.5:
        label = "Male : "
        percentage = pred[0][0] * 100.0
        label += str(round(percentage, 2))
        textColor = (255, 0, 0)
    else:
        label = "Female : "
        percentage = pred[0][1] * 100.0
        label += str(round(percentage, 2))
        textColor = (0, 0, 255)
    cv.putText(img, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2)
    # cv.imshow("gender detection", img)

cv.imwrite('output.jpg', img)
-------------------------------------------------------------------------------------------------------------------------



#
# from tkinter import *
# from PIL import ImageTk, Image
# root = Tk()
# root.title('Title')
# img = ImageTk.PhotoImage(Image.open("test.jpg"))
# label = Label(root, image = img)
# label.pack()
# root.mainloop()
