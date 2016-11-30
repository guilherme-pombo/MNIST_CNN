import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras.utils.np_utils as kutils
import predict
import numpy as np

# Number of pixels per image, 28x28
IMG_ROWS, IMG_COLS = 28, 28
# Normalisation constant to convert training data to [0,1] range
NORM_CONS = 255.0
# Number of epochs to generate data in
EPOCHS = 50
# Amount of batches per epoch
BATCH_PER_EPOCH = 20


'''
This method is used for data augmentation by applying transforms to the existing data, hence creating
new "artificial" data
  - Pass in visualize=True to generate only a few images and visualize their transformations
  - Otherwise, use data augmentation and train the model with the new augmented data
'''


def data_augment(visualize=False):
    # Load in data from files
    train_pixels, train_labels, test_pixels, num_classes = predict.read_data()

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=40,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=False)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(train_pixels)

    # If we want to visualize, generate a few images and visualize them to see if the transformations
    # are good
    if visualize:
        # Generate artificial images, by transforming the existing ones
        for e in range(1):
            batches = 0
            for X_batch, Y_batch in datagen.flow(train_pixels, train_labels, batch_size=1):
                print X_batch.shape
                print Y_batch
                display(X_batch)
                batches += 1
                if batches >= 6:
                    break

    # Generate the model and train it the augmented data
    else:
        model = predict.create_simple_model(num_classes)
        # fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen.flow(train_pixels, train_labels, batch_size=32),
                            samples_per_epoch=len(train_pixels), nb_epoch=EPOCHS)
        predictions = model.predict_classes(test_pixels)
        np.savetxt('augmented_submission.csv', np.c_[range(1, len(predictions) + 1), predictions],
                   delimiter=',', header='ImageId,Label', fmt='%d')


'''
Auxiliary function to display an image
Just pass in train_pixels[x]
'''


def display(img):
    one_image = img.reshape(IMG_ROWS, IMG_COLS)
    plt.axis('on')
    plt.imshow(one_image, cmap=plt.cm.binary)
    plt.show()


if __name__ == '__main__':
    data_augment()
