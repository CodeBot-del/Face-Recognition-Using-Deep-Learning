import cv2 
import numpy as np 
import face_recognition 
import os
import pandas as pd

path = 'imagesRecognition'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

#import the images from the directory
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    
print(classNames)

#function to encode all images in the directory
def findEncodings(images):
    encodeList = [] 
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] 
        encodeList.append(encode)
    return encodeList
                                         
encodeListKnown = findEncodings(images)  
print('################------ Encoding Complete ------################') 
print(encodeListKnown)
encodedFaces = encodeListKnown

#save with numpy savetxt
# np.savetxt('encodings.csv',encodeListKnown, delimiter=',' ) ## bad idea

#save with pandas dataframe
# df = pd.DataFrame(encodeListKnown)
# df.to_csv('encodings.csv')    # Not soo bad

#Save in a text file.....BEST SOLUTION

# new_array = encodeListKnown

# #open the text file
# file = open("encodings.txt", "w+")

# #saving the array in the text file 
# content = str(new_array)
# file.write(content)
# file.close()

# #Display the contents of the text file
# file = open("encodings.txt", "r")
# content = file.read()

# print("The encodings saved are: ", content)
# file.close()
