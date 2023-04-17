from pathlib import Path
from io import BytesIO
import cv2
from PIL import Image
import numpy as np
from keras import models
from keras.utils import load_img, img_to_array, array_to_img

model = models.load_model(Path(__file__).parent / 'model.h5')
# noinspection PyUnresolvedReferences
vid = cv2.VideoCapture(0)
# print([i for i in dir(cv2) if 'CAP' in i])
# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
face_cascade = cv2.CascadeClassifier(str(Path(__file__).parent / 'haarcascade_frontalface_default.xml'))

ret, frame = vid.read()
while ret:
    ret, frame = vid.read()
    # noinspection PyUnresolvedReferences
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4, minSize=(40, 40))
    for x, y, w, h in faces:
        # noinspection PyUnresolvedReferences
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # noinspection PyUnresolvedReferences
    cv2.imshow('frame', frame)
    # img = load_img(frame, target_size=(48, 48))
    #
    img = array_to_img(frame).resize((48, 48))
    with BytesIO() as byte_io:
        img.save(byte_io, format='png')
        img = load_img(byte_io, target_size=(48, 48, 3))
        img_arr = img_to_array(img)
        print(model.predict([img_arr]))

    # noinspection PyUnresolvedReferences
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break

vid.release()
# noinspection PyUnresolvedReferences
cv2.destroyAllWindows()
