import json
import numpy as np
from sklearn.model_selection import train_test_split #split data into train and test
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_PATH = "genre.json"
USER_PATH = "user_data.json"

def load_data(data_path):


    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def prepare_datasets(test_size, validation_size):
    #load the data
    X, y = load_data(DATA_PATH)
    #create train and test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= test_size) #percent for test
    #create the train and  validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size) #percent for train
    X_train = X_train[..., np.newaxis] #4d array -> (num_samples, 130, 12, 1)
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    #create model
    model = keras.Sequential()
    #1st layer
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu', input_shape = input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding = 'same' ))
    model.add(keras.layers.BatchNormalization())
    #2nd layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    #3rd layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    #flatten output and feed into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    #output layer
    model.add(keras.layers.Dense(4, activation='softmax')) #scores for the 4 categories/neurons/genres

    return model
def predict(model,X,y):
    X = X[np.newaxis, ...]
    predictions = model.predict(X)
    predicted_index = np.argmax(predictions, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y,predicted_index))

if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2) #test size, validation size
    # build the CNN sets
    input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
    model = build_model (input_shape)
    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    # train the CNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=20)
    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose = 1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    #prediction on a sample
    X = X_test[11]
    y = y_test[11] #change these 2 values to change which sample to test
    predict(model,X, y)
    #plotting accuracy vs value accuracy over epoch
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    #plotting confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['angry', 'happy', 'relaxed', 'sad']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.show()
