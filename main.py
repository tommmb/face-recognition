import os.path

import cv2.cv2 as cv2
import cv2.data as data
import numpy as np
import pickle
import time

face_cascade = cv2.CascadeClassifier(data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
labels = {}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {value: key for key, value in og_labels.items()}

# {'peter-dinklage': 0, 'kit-harington': 1, 'justin': 2}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y: y + h, x: x + h]
        roi_color = frame[y: y + h, x: x + h]

        id_, conf = recognizer.predict(roi_gray)
        name = labels[id_]

        if conf <= 30:
            # print(f'{labels[id_]} with {conf} confidence')
            num_faces = len(os.listdir(os.path.join(os.getcwd(), 'faces/tom-burke')))
            label_path = os.path.join(os.getcwd() + f'/faces/{name}/')
            file_path = os.path.join(label_path, f'{num_faces}.jpg')
            cv2.imwrite(file_path, roi_color)
            time.sleep(2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        # base_dir = os.path.dirname(os.path.abspath(__file__))
        # if os.path.isdir(label_path):
        #     cv2.imwrite(label_path, roi_color)
        # else:
        #     print(f'{label_path} is not a directory')

        # img_item = 'my-item-color.png'
        # cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    cv2.imshow('img', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()