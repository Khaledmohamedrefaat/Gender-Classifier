# Step 5 - Detect Input Camera Frames
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.utils import get_file
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