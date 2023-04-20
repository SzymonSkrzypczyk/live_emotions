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


def get_colour(_emotion: str):
    match _emotion:
        case "Angry":
            return 0, 0, 255
        case "Disgust":
            return 255, 0, 191
        case "Fear":
            return 128, 128, 128
        case "Happy":
            return 0, 255, 64
        case "Sad":
            return 255, 0, 0
        case "Surprise":
            return 0, 255, 255
        case "Neutral":
            return 255, 255, 0


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
        img = Image.fromarray(frame).crop((x, y, x + w, y + h))
        # add coords!!!
        # img = array_to_img(frame).resize((48, 48))
        with BytesIO() as byte_io:
            img.save(byte_io, format='png')
            img = load_img(byte_io, target_size=(48, 48))
            img_arr = img_to_array(img)
            prediction = model.predict(np.expand_dims(img_arr, axis=0), verbose=False)
            prediction.flatten()
            predicted_text = f'{EMOTIONS[int(np.argmax(prediction))]}: {np.max(prediction):.4f}'
            col = get_colour(EMOTIONS[int(np.argmax(prediction))])
            # noinspection PyUnresolvedReferences
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            # noinspection PyUnresolvedReferences
            cv2.putText(
                frame, predicted_text, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2, cv2.LINE_AA
            )
    # noinspection PyUnresolvedReferences
    cv2.imshow('frame', frame)
    # noinspection PyUnresolvedReferences
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break

vid.release()
# noinspection PyUnresolvedReferences
cv2.destroyAllWindows()
