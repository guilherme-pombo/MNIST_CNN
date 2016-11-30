import pandas as pd
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss

'''
This was all done with Keras using Theano backend. Haven't tried Tensorflow for performance yet.
However, it is easy to change by nano ~/.keras/keras.json
and changing the backend param
'''

# Amount of data that the CNN takes in at a time -- bigger batch size, more RAM
BATCH_SIZE = 128
# Number of pixels per image, 28x28
IMG_ROWS, IMG_COLS = 28, 28
# Normalisation constant to convert training data to [0,1] range
NORM_CONS = 255.0

'''
Load in training and testing data and then reshape so it is in [0,1] range
rather than in [0,255] range
There are 10 different classes (10 different digits)
'''


def read_data():
    # Load in data from files
    train = pd.read_csv("train.csv").values
    test = pd.read_csv("test.csv").values
    # Reshape data, taking into account it comes in 28x28 blocks
    # Train
    train_pixels = train[:, 1:].reshape(train.shape[0], 1, IMG_ROWS, IMG_COLS)
    train_pixels = train_pixels.astype(float)
    train_pixels /= NORM_CONS
    # Test
    test_pixels = test.reshape(test.shape[0], 1, 28, 28)
    test_pixels = test_pixels.astype(float)
    test_pixels /= NORM_CONS
    # Get labels
    train_labels = kutils.to_categorical(train[:, 0])
    num_classes = train_labels.shape[1]
    return train_pixels, train_labels, test_pixels, num_classes


'''
Use this method to train a simple Convolutional neural net that has two layers with:
  - 32 filters first layer
  - 64 filters second layer
Only runs for 5 epochs and achieves around 96% accuracy on the leaderboard.
More easily adaptable to other competitions given it's simpler structure.
Similar to Keras tutorial on VGG net
'''


def create_simple_model(num_classes, layer1_filters=32, layer2_filters=64):
    epochs = 5
    n_conv = 2
    model = models.Sequential()

    # First layer
    model.add(conv.ZeroPadding2D((1, 1), input_shape=(1, IMG_COLS, IMG_ROWS),))
    model.add(conv.Convolution2D(layer1_filters, n_conv, n_conv,  activation="relu"))
    model.add(conv.MaxPooling2D(strides=(2, 2)))

    # Second layer
    model.add(conv.ZeroPadding2D((1, 1)))
    model.add(conv.Convolution2D(layer2_filters, n_conv, n_conv, activation="relu"))
    model.add(conv.MaxPooling2D(strides=(2, 2)))

    model.add(core.Flatten())
    model.add(core.Dropout(0.2))
    model.add(core.Dense(128, activation="relu"))
    model.add(core.Dense(num_classes, activation="softmax"))

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    return model, epochs


'''
Use this method to train a more complex Convolutional neural net that has two three with:
  - 64 filters first layer
  - 128 filters second layer
  - 256 filters third layer
Runs for 100 epochs and achieves around 99% accuracy on the test data. Works good for Kaggle, for general purposes
it would be good to be careful with using this many epochs without callback (i.e. using a validation set to stop
the training early in case the model starts overfitting)
Similar to the VGGnet model
'''


def create_complex_model(num_classes):
    epochs = 100
    layer1_filters = 64
    layer2_filters = 128
    layer3_filters = 256
    n_conv = 3
    model = models.Sequential()

    # First layer
    model.add(conv.ZeroPadding2D((1, 1), input_shape=(1, IMG_ROWS, IMG_COLS), ))
    model.add(conv.Convolution2D(layer1_filters, n_conv, n_conv, activation="relu"))
    model.add(conv.ZeroPadding2D((1, 1)))
    model.add(conv.Convolution2D(layer1_filters, n_conv, n_conv, activation="relu"))
    model.add(conv.MaxPooling2D(strides=(2, 2)))

    # Second layer
    model.add(conv.ZeroPadding2D((1, 1)))
    model.add(conv.Convolution2D(layer2_filters, n_conv, n_conv, activation="relu"))
    model.add(conv.ZeroPadding2D((1, 1)))
    model.add(conv.Convolution2D(layer2_filters, n_conv, n_conv, activation="relu"))
    model.add(conv.MaxPooling2D(strides=(2, 2)))

    # Third layer
    model.add(conv.ZeroPadding2D((1, 1)))
    model.add(conv.Convolution2D(layer3_filters, n_conv, n_conv, activation="relu"))
    model.add(conv.ZeroPadding2D((1, 1)))
    model.add(conv.Convolution2D(layer3_filters, n_conv, n_conv, activation="relu"))
    model.add(conv.ZeroPadding2D((1, 1)))
    model.add(conv.Convolution2D(layer3_filters, n_conv, n_conv, activation="relu"))
    model.add(conv.ZeroPadding2D((1, 1)))
    model.add(conv.Convolution2D(layer3_filters, n_conv, n_conv, activation="relu"))
    model.add(conv.MaxPooling2D(strides=(2, 2)))

    model.add(core.Flatten())
    model.add(core.Dropout(0.2))
    model.add(core.Dense(4096, activation="relu"))
    model.add(core.Dense(num_classes, activation="softmax"))

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    return model, epochs

'''
Use this method to actually fit the model and create the submission file
'''


def fit_model(model, train_pixels, train_labels, test_pixels, epochs):
    model.fit(train_pixels, train_labels, batch_size=BATCH_SIZE, nb_epoch=epochs, verbose=1)

    predictions = model.predict_classes(test_pixels)
    np.savetxt('submission.csv', np.c_[range(1, len(predictions) + 1), predictions],
               delimiter=',', header='ImageId,Label', fmt='%d')


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


'''
Use this method to do a k-fold cross validation on the simple model (can be done with the complex model)
Use it to test number of filters to use on each layer (can be extended to test number of epochs, etc.)
'''


def run_cross_validation(nfolds=10, layer1_filters=32, layer2_filters=64):
    train_pixels, train_labels, test_pixels, num_classes = read_data()
    yfull_train = dict()
    yfull_test = []
    kf = KFold(len(train_pixels), n_folds=nfolds, shuffle=True)
    num_fold = 0
    for train_index, test_index in kf:
        model, epochs = create_simple_model(num_classes, layer1_filters, layer2_filters)
        X_train, X_valid = train_pixels[train_index], train_pixels[test_index]
        Y_train, Y_valid = train_labels[train_index], train_labels[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=epochs, shuffle=True, verbose=2,
                  validation_data=(X_valid, Y_valid), callbacks=callbacks)

        predictions_valid = model.predict(X_valid, batch_size=BATCH_SIZE, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        # Store test predictions
        test_prediction = model.predict(test_pixels, batch_size=BATCH_SIZE, verbose=2)
        yfull_test.append(test_prediction)

    predictions_valid = get_validation_predictions(train_pixels, yfull_train)
    score = log_loss(train_labels, predictions_valid)
    print("Log_loss train independent avg: ", score)

    print('Final log_loss: {}, nfolds: {}'.format(score, nfolds))

if __name__ == '__main__':
    train_pixels, train_labels, test_pixels, num_classes = read_data()
    complex_model, epochs = create_complex_model(num_classes)
    fit_model(complex_model, train_pixels, train_labels, test_pixels, epochs)
