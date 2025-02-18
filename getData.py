import os
import numpy as np
import cv2
import skimage

read_directory = os.path.join(os.path.abspath(os.getcwd()), r'new_faces')
save_directory = os.path.abspath(os.getcwd())
id2img = []
id2Label = []
img_Id = 0
dataSet = []    #list of tuple[img1_id,img2_id]
labelSet = []    #list of label

with os.scandir(read_directory) as entries:
    for entry in entries:
        if not entry.name.endswith(".DS_Store"):
            os.chdir(entry.path)
#            print(entry.name)
            for filename in os.listdir(entry.path):
#                if img_Id >=1:
#                    break
                image = cv2.imread(filename,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,(57,77))
                image = skimage.img_as_float32(image)
                id2img.append(image)
                id2Label.append(entry.name)
                img_Id+=1
#                os.chdir('/Users/rasn/Downloads')
#                cv2.imwrite('test.jpg', image)
t_n = 0
f_n = 0
for i in range(img_Id):
    for j in range(i+1,img_Id):
            label = id2Label[i]==id2Label[j]
            if label:
                if t_n<4000:
                    t_n+=1
                else:
                    continue
            else:
                if f_n<4000:
                    f_n+=1
                else:
                    continue
            data = [i,j,label]
            dataSet.append(data)
dataSet = np.array(dataSet)
np.random.shuffle(dataSet)
print(t_n,f_n,t_n+f_n)
print(dataSet.shape)
os.chdir(save_directory)
train_data = dataSet[0:int(0.8*dataSet.shape[0]),0:2]
train_label = dataSet[0:int(0.8*dataSet.shape[0]),2:3]
test_data = dataSet[int(0.8*dataSet.shape[0]):,0:2]
test_label = dataSet[int(0.8*dataSet.shape[0]):,2:3]
print(train_data.shape,train_label.shape,test_data.shape,test_label.shape)
np.save('img_db.npy', id2img)
np.save('train_data.npy', train_data)
np.save('train_labels.npy', train_label)
np.save('test_data.npy', test_data)
np.save('test_labels.npy', test_label)
