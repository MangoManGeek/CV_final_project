#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2019 Iván de Paz Centeno
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import time
import numpy as np
import cv2
from mtcnn import MTCNN
import emotion_detection as ed

cap = cv2.VideoCapture(0)
detector = MTCNN()
time_start = time.time()
time_interval = 0.1
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

while(True):
#    time.sleep(0.05)
    # Capture frame-by-frame
    ret, original_image = cap.read()
    image = original_image
    # Our operations on the frame come here
    if time.time()-time_start>=time_interval:
        result = detector.detect_faces(original_image)
        for r in result:
            if 'box' in r and 'keypoints' in r:
                image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                bounding_box = r['box']
                keypoints = r['keypoints']

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
            emotion = ed.predict_emotion(original_image)
            cv2.putText(image,emotion, (bounding_box[0], bounding_box[1]), font, fontScale,fontColor,lineType)
            time_start = time.time()
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

