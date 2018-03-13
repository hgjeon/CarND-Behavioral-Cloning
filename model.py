import os
import csv
import cv2
import numpy as np
import sklearn

data_directory = 'data'
fname_save_model = 'model'

#data_directory = 'data_3'
#data_directory = 'data_2_recovery'
#fname_load_model = 'model_save_01'
#fname_save_model = 'model_save_02'

samples = []
with open(data_directory+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import random

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

### Generator function for memory utilization
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:    # Loop forever
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = data_directory+'/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                if (np.random.randint(0,2)):
                    images.append(center_image)
                    angles.append(center_angle)
                    #print('N')
                else:
                    # Flip the image
                    image_flipped = np.fliplr(center_image)
                    images.append(image_flipped)
                    angles.append(-center_angle)
                    #print('R')

            #print("Images, Angles = ", images.shape, angles.shape)

            # Trip image here
            X_train = np.array(images)
            y_train = np.array(angles)

            #print("X_train, y_train = ", X_train.shape, y_train.shape)

            yield sklearn.utils.shuffle(X_train, y_train)

max_x = 320
max_y = 160
crop_y_top = int(max_y * 32 / 100)
crop_y_bottom = int(max_y * 13 / 100)
#print ("crop_y_top, bottom = ", crop_y_top, crop_y_bottom)

ch, row, col = 3, 160, 320

# Compile and Train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

### Model architecture

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.20))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.20))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.20))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.20))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.20))
model.add(Dense(50))
model.add(Dropout(0.20))
model.add(Dense(10))
model.add(Dropout(0.20))
model.add(Dense(1))

### Load weights from previous training
#model.load_weights(fname_load_model+'_weight.h5')

### Train model
model.compile(loss='mse', optimizer='adam')

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
#history_obj = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)
history_obj = model.fit_generator(train_generator,
                samples_per_epoch=(len(train_samples) * 1),
                validation_data=validation_generator,
                nb_val_samples=(len(validation_samples) * 1), nb_epoch=5)

model.save_weights(fname_save_model+'_weight.h5')
model.save(fname_save_model+'.h5')

### Print the keys contained in the history object
print(history_obj.history.keys())

plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('Model MSE loss')
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.legend(['training set', 'validation set'])
plt.show()
