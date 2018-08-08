import csv
import cv2
import numpy as np
import random
from keras.layers import ELU

def generator(features, labels, batch_size):

 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, 160, 320, 3))
 batch_labels = np.zeros((batch_size,1))
 print(len(features))
 while True:
   for i in range(batch_size):
     index = random.choice(list(range(len(features))))
     #Here we have a one in four chance to flip an image.
     #This helps fix the left turn bias the model has
     if(index % 4 ==  0):
         flippedImage = np.fliplr(features[index])
         flippedMeasurement = -labels[index]
     else:
         flippedImage = (features[index])
         flippedMeasurement = labels[index]
     batch_features[i] = (flippedImage)
     batch_labels[i] = (flippedMeasurement)
   yield batch_features, batch_labels


lines = []
with open('sample_driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
minval = 0.020 #This value makes it so that all images with an associated
#driving angle of less than 0.5 will be removed from the training set


#Here we get the images from the Udacity training data
for line in lines:
    for column in line[:2]:
        source_path = column
        filename = source_path.split('/')[-1]
        current_path = 'sampleIMG/' + filename
        measurement = float(line[3])
        if (abs(measurement) < minval):
            pass
        else:
            measurements.append(measurement)
            image = cv2.imread(current_path)
            images.append(image)

with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Here we get the images from personal training sessions
for line in lines:
    for column in line[:2]:
        source_path = column
        filename = source_path.split('/')[-1]
        current_path = 'IMG/' + filename

        measurement = float(line[3])
        # measurements.append(measurement)
        if (abs(measurement) < minval):
            pass
        else:
            measurements.append(measurement)
            image = cv2.imread(current_path)
            images.append(image)

print(len(images))
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()

# First we crop the image then put it through a number of convolutional layers
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160,320,3)))
model.add(Conv2D(3, (5, 5),activation = "relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(24, (5, 5),activation = "relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(36, (5, 5),activation = "relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(48, (3, 3),activation = "relu"))
model.add(MaxPooling2D(2,2))

#Here we combine normal layers and dropout layers with ELU activation functions
model.add(Flatten())
model.add(Dropout(0.25))
model.add(ELU())
model.add(Dense(800))
model.add(Dropout(0.25))
model.add(ELU())
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(ELU())
model.add(Dense(50))
model.add(Dropout(0.25))
model.add(ELU())
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(generator(images, measurements, batch_size = 32), steps_per_epoch=600, nb_epoch = 75)

# model.fit(X_train, y_train, validation_split=0.2,shuffle=True, epochs = 3,batch_size = 6)
print(model.summary())
model.save('finalModel.h5')
