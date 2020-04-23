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

import numpy as np
import cv2
from mtcnn import MTCNN

cap = cv2.VideoCapture(0)
detector = MTCNN()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    for r in result:
        if 'box' in r and 'keypoints' in r:
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
    
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

