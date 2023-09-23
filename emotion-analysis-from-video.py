import keras.models
import numpy as np
import cv2
from keras.preprocessing import image
import tensorflow as tf
from keras.models import model_from_json

# -----------------------------
# opencv initialization

face_cascade_default = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# -----------------------------
# face expression recognizer initialization

# model = model_from_json(open("./facial_expression_model_structure.json", "r").read())
# model.load_weights('./facial_expression_model_weights.h5')  # load weights
model = keras.models.load_model("./ferModel")

# -----------------------------

emotions = ('anger', 'happiness', 'neutral', 'sadness', 'surprise-or-fear')

# cap = cv2.VideoCapture('test.mp4') #process videos
cap = cv2.VideoCapture(0)  # process real time web-cam
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FOURCC, 0x32595559)
# cap.set(cv2.CAP_PROP_FPS, 25)

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (640, 360))

    if ret is True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade_default.detectMultiScale(img, 1.2, 5)

        for (x, y, w, h) in faces:
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_GRAY2BGR)
            detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48
            detected_face = cv2.resize(detected_face, (75, 75))

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            # ------------------------------

            predictions = model.predict(img_pixels)  # store probabilities of 7 expressions
            max_index = np.argmax(predictions[0])

            # Draw rect around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

            emotion = "%s %s%s" % (emotions[max_index], round(predictions[0][max_index] * 100, 2), '%')

            """if i != max_index:
                color = (255,0,0)"""

            cv2.putText(img, emotion, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)

        # -------------------------

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

# kill open cv things
cap.release()
cv2.destroyAllWindows()
