from pathlib import Path
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from keras import models
from keras.utils import load_img, img_to_array, array_to_img

EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

model = models.load_model(Path(__file__).parent / 'model.h5')
# noinspection PyUnresolvedReferences
vid = cv2.VideoCapture(0)
# print([i for i in dir(cv2) if 'CAP' in i])
# noinspection PyUnresolvedReferences
face_cascade = cv2.CascadeClassifier(str(Path(__file__).parent / 'haarcascade_frontalface_default.xml'))

ret, frame = vid.read()
while ret:
    ret, frame = vid.read()
    # noinspection PyUnresolvedReferences
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4, minSize=(48, 48))
    images = []
    for x, y, w, h in faces:
        # noinspection PyUnresolvedReferences
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        images.append((Image.fromarray(frame).crop((x, y, x + w, y + h)), x + w, y + h))
    # noinspection PyUnresolvedReferences
    cv2.imshow('frame', frame)
    # img = load_img(frame, target_size=(48, 48))

    for i in images:
        img = i[0]
        # add coords!!!
        # img = array_to_img(frame).resize((48, 48))
        with BytesIO() as byte_io:
            img.save(byte_io, format='png')
            img = load_img(byte_io, target_size=(48, 48))
            img_arr = img_to_array(img)
            prediction = model.predict(np.expand_dims(img_arr, axis=0), verbose=False)
            prediction.flatten()
            print(EMOTIONS[int(np.argmax(prediction))])
            predicted_text = f'{EMOTIONS[int(np.argmax(prediction))]}: {prediction[0]}'
            # noinspection PyUnresolvedReferences
            cv2.putText(
                frame, predicted_text, (i[1], i[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            )

    # noinspection PyUnresolvedReferences
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break

vid.release()
# noinspection PyUnresolvedReferences
cv2.destroyAllWindows()
