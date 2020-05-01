import os
import numpy as np
import cv2
directory = r'/Users/rasn/GitHub/CV_final_project/new_faces'
id2img = []
id2Label = []
img_Id = 0
dataSet = []    #list of tuple[img1_id,img2_id]
labelSet = []    #list of label

with os.scandir(directory) as entries:
    for entry in entries:
        if not entry.name.endswith(".DS_Store"):
            os.chdir(entry.path)
#            print(entry.name)
            for filename in os.listdir(entry.path):
#                if img_Id >=1:
#                    break
                image = cv2.imread(filename,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,(230,310))
                id2img.append(image)
                id2Label.append(entry.name)
                img_Id+=1
#                os.chdir('/Users/rasn/Downloads')
#                cv2.imwrite('test.jpg', image)
for i in range(img_Id):
    for j in range(i+1,img_Id):
            label = id2Label[i]==id2Label[j]
            data = [i,j,label]
            dataSet.append(data)
dataSet = np.array(dataSet)
np.random.shuffle(dataSet)
print(dataSet.shape)
os.chdir(r'/Users/rasn/GitHub/CV_final_project/')
train_data = dataSet[0:int(0.8*dataSet.shape[0]),0:2]
train_label = dataSet[0:int(0.8*dataSet.shape[0]),2:3]
test_data = dataSet[int(0.8*dataSet.shape[0]):,0:2]
test_label = dataSet[int(0.8*dataSet.shape[0]):,2:3]
print(train_data.shape,train_label.shape,test_data.shape,test_label.shape)
np.save('img_db', id2img)
np.save('train_data', train_data)
np.save('train_label', train_label)
np.save('test_data', test_data)
np.save('test_label', test_label)