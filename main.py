from pathlib import Path
from keras import models
from keras import layers
from keras.utils import image_dataset_from_directory
import pandas as pd
"""
0 - Angry
1 - Disgust
2 - Fear
3 - Happy
4 - Sad
5 - Surprise
6 - Neutral
"""

TEST_IMAGES = Path(__file__).parent / 'images' / 'test'
TRAIN_IMAGES = Path(__file__).parent / 'images' / 'train'
VAL_IMAGES = Path(__file__).parent / 'images' / 'val'


def get_data():
    train = image_dataset_from_directory(
        TRAIN_IMAGES,
        label_mode='categorical',
        # class_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
        image_size=(48, 48)
    )
    val = image_dataset_from_directory(
        VAL_IMAGES,
        label_mode='categorical',
        # class_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
        image_size=(48, 48)
    )
    test = image_dataset_from_directory(
        TEST_IMAGES,
        label_mode='categorical',
        # class_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
        image_size=(48, 48)
    )
    return train, val, test


def create_model(_train, _val, _epochs: int = 10, _batch_size: int = 32):
    _model = models.Sequential()
    _model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
    _model.add(layers.MaxPooling2D((2, 2)))
    _model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    _model.add(layers.MaxPooling2D((2, 2)))
    _model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    _model.add(layers.GlobalMaxPooling2D())
    _model.add(layers.Flatten())
    _model.add(layers.Dense(128, activation='relu'))
    _model.add(layers.Dropout(.55))
    _model.add(layers.Dense(64, activation='relu'))
    _model.add(layers.Dense(7, activation='softmax'))
    _model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='categorical_crossentropy')
    _model.fit(_train, epochs=_epochs, validation_data=_val, batch_size=_batch_size)
    return _model


if __name__ == '__main__':
    data = get_data()
    # model = create_model(data[0], data[1], _epochs=5)
    model = models.load_model('model.h5')
    print(model.predict(data[2]))
