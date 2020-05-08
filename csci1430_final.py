#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import cv2
from mtcnn import MTCNN
import emotion_detection as ed

cap = cv2.VideoCapture(0)
detector = MTCNN()
time_start = time.time()
time_interval = 1
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
emotion                = "None"
while(True):
#    time.sleep(0.05)
    # Capture frame-by-frame
    ret, image = cap.read()
    # Our operations on the frame come here
    if time.time()-time_start>=time_interval:
        result = detector.detect_faces(image)
        for r in result:
            if 'box' in r and 'keypoints' in r:
                bounding_box = r['box']
                keypoints = r['keypoints']
                face_img = image[bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.rectangle(image,
                                        (bounding_box[0], bounding_box[1]),
                                        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                                        (0,155,255),
                                        2)
                cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # Display the resulting frame
            emotion = ed.predict_emotion(face_img)
#            cv2.putText(image,emotion, (bounding_box[0], bounding_box[1]), font, fontScale,fontColor,lineType)
            time_start = time.time()
    if emotion!="None":
        cv2.putText(image,emotion, (bounding_box[0], bounding_box[1]), font, fontScale,fontColor,lineType)
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

