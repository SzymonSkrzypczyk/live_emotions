from pathlib import Path
import cv2
# noinspection PyUnresolvedReferences
vid = cv2.VideoCapture(0)
# noinspection PyUnresolvedReferences
# cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# noinspection PyUnresolvedReferences
face_cascade = cv2.CascadeClassifier(str(Path(__file__).parent / 'haarcascade_frontalface_default.xml'))

while True:
    ret, frame = vid.read()

    # noinspection PyUnresolvedReferences
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4, minSize=(40, 40))
    for x, y, w, h in faces:
        # noinspection PyUnresolvedReferences
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # noinspection PyUnresolvedReferences
    cv2.imshow('frame', frame)

    # noinspection PyUnresolvedReferences
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break
vid.release()
# noinspection PyUnresolvedReferences
cv2.destroyAllWindows()
