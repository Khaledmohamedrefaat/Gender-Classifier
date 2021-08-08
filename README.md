# Gender Classifier: A Deep Learning Male/Female Classifier

![Gender Classifier Logo](https://user-images.githubusercontent.com/25768661/128627699-8a60f06d-8719-4a64-a2c1-9222cdc36a13.png)

# What is Gender Classifier? 
Gender Classifier is python program which shows the functionality of a deep learning convolutional neural network that classifies between male and females.

## Table of contents
* [Technologies](#technologies)
  * [Dataset](#dataset)
  * [Model](#model)
  * [Setup](#setup)
  * [Features](#features)
    + [Use Still Image](#use-still-image)
    + [Use Live Camera](#use-live-camera)
  * [Contributers](#contributers)

## Technologies
* Programming Language: Python 
* GUI Framework: Tkinter
* Deep Learning: CNNs

## Dataset
* The model was trained on a data set which was a combination between alot of other datasets on the internet, The dataset included :
  * 5000 Male Faces.
  * 5000 Female Faces.

* A link to the dataset will be uploaded soon.
## Model
The model consisted of 7 layers:
* Layer 1 - Convolution-Normalizing-MaxPooling - Dropout 0.25
* Layer 2 - Convolution-Convolution-Normalizing-MaxPooling - Dropout 0.25
* Layer 3 - Convolution-Convolution-Normalizing-MaxPooling - Dropout 0.25
* Layer 4 - Convolution-Convolution-Normalizing-MaxPooling - Dropout 0.25
* Layer 5 - Flattening
* Layer 6 - FCC Input Layer - Dropout 0.5
* Layer 7 - FCC Output Layer - Activation Softmax

The model was trained for 200 epochs with 32 batches and got the following results:
* 97% Training Accuracy
* 93% Testing Accuracy

## Setup

To run the python application you need to install the requirements (It is preffered to start a new environment and install all the requirements) using the following command

```bash
 pip install -r requirements.txt
```



## Features
### Use Still Image
Browse and select any image, The program will automatically detect the faces and each face will passed to the pre-trained model for classification.

The output will be automatically displayed and also saved as output.jpg in the program directory.
![Still Image](https://user-images.githubusercontent.com/25768661/128628260-2ff43600-1fba-4f48-8717-1cc643d43f1b.gif)
### Use Live Camera
Instead of using an image, If you have a camera in your PC/Laptop, you can press "Use Live Camera" to get the classification done on the camera input.

Each frame will be passed to the model for classification and the output will be automatically shown.
![index](https://user-images.githubusercontent.com/25768661/128634496-15bd98da-2a4f-45a2-924c-7342e8a85550.png)


## Contributers
[Khaled Mohamed](https://github.com/Khaledmohamedrefaat)

[Dareen Zeyad](https://github.com/DareenZeyad)

[Omnia Hosny](https://github.com/OmniaHQ)

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

