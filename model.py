import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.misc import toimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, MaxPooling2D, PReLU
from keras.layers.convolutional import Convolution2D
from drive import preprocess_image

print('Loading data...')
lines = []
with open('./my_data/driving_log.csv') as csvfile:
    has_header = csv.Sniffer().has_header(csvfile.read(1024))
    csvfile.seek(0)
    reader = csv.reader(csvfile)
    if has_header:
        next(reader)  # Skip header row.
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Quick Look at the number of data
print("Number of total Data: ", len(lines))
print("Number of Training Data: ", len(train_samples))
print("Number of Validation Data: ", len(validation_samples))

# Using Generator to load a batch of images
def generator(samples, batch_size=32):
    """
    generate a batch of data.
    """
    Corr_factor = 0.25
    num_samples = len(samples)
    while 1: 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+(batch_size)]

            images = []
            angles = []
            for batch_sample in batch_samples:
                path = './my_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = preprocess_image(cv2.imread(path))
                center_angle = float(batch_sample[3])
                path = './my_data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = preprocess_image(cv2.imread(path))
                left_angle = float(batch_sample[3])+Corr_factor
                path = './my_data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = preprocess_image(cv2.imread(path))
                right_angle = float(batch_sample[3])-Corr_factor
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

            # Augment Data by flipping
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield shuffle(X_train, y_train)

def get_my_model():
    model = Sequential()
    # normalization
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(66, 200, 3), output_shape=(66, 200, 3)))

    # convolutional and maxpooling layers
    model.add(Convolution2D(24, 5, 5, border_mode='valid',
                            subsample=(2, 2), init='he_normal'))
    model.add(PReLU())
    model.add(Convolution2D(36, 5, 5, border_mode='valid',
                            subsample=(2, 2), init='he_normal'))
    model.add(PReLU())
    model.add(Convolution2D(48, 5, 5, border_mode='valid',
                            subsample=(2, 2), init='he_normal'))
    model.add(PReLU())
    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            subsample=(1, 1), init='he_normal'))
    model.add(PReLU())
    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            subsample=(1, 1), init='he_normal'))
    model.add(PReLU())
    model.add(Flatten())

    # fully connected layers
    model.add(Dense(100, init='he_normal'))
    model.add(Dropout(0.50))
    model.add(Dense(50, init='he_normal'))
    model.add(Dropout(0.50))
    model.add(Dense(10, init='he_normal'))
    model.add(Dropout(0.50))
    model.add(Dense(1))
	
    model.summary()
    model.compile(optimizer='adam', loss="mse")
    return model


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = get_my_model()
print("Training...")
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('./model_car.h5')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig("./result.jpg")
