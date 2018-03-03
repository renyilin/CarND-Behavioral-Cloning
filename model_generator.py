import numpy as np
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Activation, MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import csv
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

print('Loading data...')
lines =[]
with open('data/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

Corr_factor = 0.2
batchSize = 32

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
def generator(samples, batch_size=batchSize):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            lines = samples[offset:offset+batch_size]

            images = []
            angles = []
            for line in lines:
				#Adding center image 
                name = 'data/data/IMG/'+line[0].split('/')[-1]
                image = cv2.imread(name)
                angle = float(line[3])
                images.append(image)
                angles.append(angle)
			    #augmentation
                images.append(cv2.flip(image,1)) # augmentation
                angles.append(angle*-1) #augmentation
                #right image
                name = 'data/data/IMG/'+line[2].split('/')[-1]
                image = cv2.imread(name)
                angle = float(line[3])-Corr_factor
                images.append(image)
                angles.append(angle)		
			    #augmentation
                images.append(cv2.flip(image,1)) # augmentation
                angles.append(angle*-1) #augmentation
                #left image
                #name = 'data/data/IMG/'+line[1].split('/')[-1]
                #image = cv2.imread(name)
                #angle = float(line[3])+Corr_factor
                #images.append(image)
                #angles.append(angle)		
			    #augmentation
                #images.append(cv2.flip(image,1)) # augmentation
                #angles.append(angle*-1) #augmentation
				
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train) 


train_generator = generator(train_samples, batch_size=batchSize)
validation_generator = generator(validation_samples, batch_size=batchSize)

#####################33		
print()

# model building
print('Building the model...')
crop_top = 50
crop_bottom = 20

model = Sequential()

#normalizing layer 
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))

# cropping layer
model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(3,160,320)))

# Convolution Layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))

#Dropout layer
model.add(Dropout(0.5))

#model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 4)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2)))

model.add(Flatten())

# Fully connected layers
model.add(Dense(582))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

#model.add(Dense(25))
#model.add(Activation('relu'))

model.add(Dense(5))
model.add(Activation('relu'))

model.add(Dense(1))

model.summary()

print('Training...')
model.compile(optimizer=Adam(0.0001), loss="mse")
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), nb_epoch=3, 
					validation_data=validation_generator, 
					nb_val_samples=len(validation_samples), verbose = 1)
			
model.save('model.h5')
