import os
import cv2
from mtcnn import MTCNN
import numpy as np

read_directory = r'/Users/rasn/Downloads/faces'
write_directory = r'/Users/rasn/Downloads/new_faces'
color = (0,155,255)
detector = MTCNN()
count = 0
os.chdir(read_directory)   
for filename in os.listdir(read_directory):
    image = cv2.imread(filename,cv2.COLOR_BGR2RGB)
    print(filename)
    if image is None:                  
        continue
    result = detector.detect_faces(image)
    if result:
        r = result[0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bounding_box = r['box']
        keypoints = r['keypoints']
        cv2.rectangle(image,
                        (bounding_box[0], bounding_box[1]),
                        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                        color,
                        1)
        cv2.circle(image,(keypoints['left_eye']), 2, color, 1)
        cv2.circle(image,(keypoints['right_eye']), 2, color, 1)
        cv2.circle(image,(keypoints['nose']), 2, color, 2)
        cv2.circle(image,(keypoints['mouth_left']), 2, color, 1)
        cv2.circle(image,(keypoints['mouth_right']), 2, color, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        crop = image[bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
    #    cv2.imshow('frame',crop)
        os.chdir(write_directory)
        cv2.imwrite(filename, crop)
        os.chdir(read_directory)
        count += 1
print("saved "+str(count)+" images")