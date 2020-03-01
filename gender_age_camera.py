# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv
from mtcnn.mtcnn import MTCNN
# from keras.applications.vgg16 import preprocess_input
from keras_vggface.utils import preprocess_input


model_path = '/home/gender_model.ckpt'
age_model_path = '/home/age_model.ckpt'
# load model
model = load_model(model_path)
age_model = load_model(age_model_path)

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

classes = ['male', 'female']
detector = MTCNN()
# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply face detection
    # face, confidence = cv.detect_face(frame)
    detres = detector.detect_faces(frame)

    # print(face)
    # print(confidence)

    # loop through detected faces
    for idx, detface in enumerate(detres):

        # get corner points of face rectangle
        bounding_box = detface['box']
        confidences = detface['confidence']
        # print('face', face)

        if bounding_box[0] < 0:
            bounding_box[0] = 1
        if bounding_box[1] < 0:
            bounding_box[1] = 1

        startX = bounding_box[0]
        endX = bounding_box[0] + bounding_box[0] + bounding_box[2]
        startY = bounding_box[1]
        endY = bounding_box[1] + bounding_box[3]

        # crop the detected face region
        face_crop = frame[bounding_box[1]:bounding_box[1] + bounding_box[3],
                    bounding_box[0]:bounding_box[0] + bounding_box[2], :]

        # if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
        #     continue

        face_crop = cv2.resize(face_crop, (224, 224))
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        face_crop = preprocess_input(face_crop)

        cv2.rectangle(frame,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        # # # apply gender detection on face
        conf = model.predict(face_crop)[0]
        # print(conf)
        # get label with max accuracy
        idx_l = np.argmax(conf)
        mxconf = np.max(conf)
        conf_age = age_model.predict(face_crop)[0]
        age_idx = np.argmax(conf_age)
        mxconf_age = np.max(conf_age)
        # if age_idx == 0 :
        #     print('age range : 1-9')
        # elif age_idx == 1 :
        #     print ('age range : 10-19')
        # elif age_idx == 2 :
        #     print('age range : 20-29')
        # elif age_idx == 3 :
        #     print('age range : 30-39')
        # elif age_idx == 4 :
        #     print('age range : 40-49')
        # elif age_idx == 5 :
        #     print('age range : 50-59')
        # elif age_idx == 6 :
        #     print('age range : above 60')

        if idx_l == 0:
            label = 'female'
            print('female')
        else:
            label = 'male'
            print('male')
        labelbox = "{}: {:.2f}%".format(label, mxconf * 100)
        cv2.putText(frame, labelbox, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
