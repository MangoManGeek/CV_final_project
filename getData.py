import os
import cv2
directory = r'/Users/rasn/GitHub/CV_final_project/new_faces'
id2img = {}
id2Label = {}
img_Id = 0
dataSet = []    #list of tuple[img1_id,img2_id]
labelSet = []    #list of label

def get_date(directory = r'/Users/rasn/GitHub/CV_final_project/new_faces'):
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
                    id2img[img_Id]=image
                    id2Label[img_Id]=entry.name
                    img_Id+=1
    #                os.chdir('/Users/rasn/Downloads')
    #                cv2.imwrite('test.jpg', image)
    print(img_Id)
    for i in range(img_Id):
        for j in range(i,img_Id):
                dataSet.append([i,j])
                label = id2Label.get(i)==id2Label.get(j)
                print(i,j,label)
                labelSet.append(label)
    return id2img,id2Label,dataSet,labelSet
    