import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import csv
import cv2
import glob

print('Loading data...')
lines = []
with open('data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
camPos = ['center', 'left', 'right']
Corr_factor = 0.2

del(lines[0])

# ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
for line in lines:
    i = 0
    # Adding center image
    path0 = line[0]   # center image path
    filename = path0.split(camPos[i])[-1]  # extract filename(without camPos)
    path = 'data/data/IMG/center' + filename
    image = [cv2.imread(path) for path in glob.glob(path)]
    images.append(image[0])
    measurement = float(line[3])   # steering angle
    measurements.append(measurement)

    # adding left camera
    # i=1
    #path0 = line[i]
    #filename = path0.split(camPos[i])[-1]
    #path =  'data/data/IMG/left' + filename
    #image = [cv2.imread(path) for path in glob.glob(path)]
    # images.append(image[0])
    #measurement = float(line[3])+Corr_factor
    # measurements.append(measurement)

    # adding right camera
    i = 2
    path0 = line[i]
    filename = path0.split(camPos[i])[-1]
    path = 'data/data/IMG/right' + filename
    image = [cv2.imread(path) for path in glob.glob(path)]
    images.append(image[0])
    measurement = float(line[3])-Corr_factor  # ammend steering angle
    measurements.append(measurement)


##########################
##########################

# 33
print()
# print(filename)
# print(path)

X_train = np.array(images)
y_train = np.array(measurements)
print("images dimension:", X_train.shape)
print("Number of images:", len(X_train))
print("Number of labels:", len(y_train))

# augmentation
augm_img, augm_y = [], []
for img, y in zip(images, measurements):
    augm_img.append(img)
    augm_y.append(y)
    augm_img.append(cv2.flip(img, 1))
    augm_y.append(y*-1.0)
X_train = np.array(augm_img)
y_train = np.array(augm_y)
print("images dimension:", X_train.shape)
print("Number of images:", len(X_train))
print("Number of labels:", len(y_train))

# model building
print('Building the model...')
crop_top = 50
crop_bottom = 20

model = Sequential()

# normalizing the input
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)),
                     input_shape=(3, 160, 320)))

# Convolution Layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))

# Dropout Layer
model.add(Dropout(0.5))

#model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 4)))
# model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2)))

model.add(Flatten())

# Fully connected layers
model.add(Dense(582))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

# model.add(Dense(25))
# model.add(Activation('relu'))

model.add(Dense(5))
model.add(Activation('relu'))

model.add(Dense(1))

model.summary()

print('Training...')
model.compile(optimizer=Adam(0.0001), loss="mse")
model.fit(X_train, y_train, validation_split=0.2,
          shuffle=True, epochs=5, verbose=1)

model.save('model.h5')
