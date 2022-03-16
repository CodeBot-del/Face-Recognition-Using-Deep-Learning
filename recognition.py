import cv2 
import numpy as np 
import face_recognition 
import os   

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



#get the image to match with... For now we read an image, later on we will read from webcam (cv2.VideoCapture())

cap = cv2.imread('cena.jpeg')  

while True:
    # success, img = cap.read() #this line to be used when reading from webcam
    img = cap  
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) #compress the image to improve performance
    imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    
    facesCurFrame = face_recognition.face_locations(imgS) #find location of all faces in the image
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) #find encodings of all the faces in the image
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): #this grabs both the encodings and the locations of the faces in the current frame
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  #compare the image encodings with the known encodings
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  #Find how much the faces differ between the image and the known
        # print(faceDis)
        matchIndex = np.argmin(faceDis)  #take the lowest face distance to be the match 
        if matchIndex < 0.5:    # accept match index lower than 0.5... for the sake of accuracy
            matchIndex = matchIndex
        
        #hii code apa chini ni ya kuchora zile boxes za green kwa wwatu verified, na red kwa wasiokua verified
        
        if matches[matchIndex]:
            
            
            name = classNames[matchIndex].upper()  #Get the Name of the image(person) that matched successfully
            # print(name)
            y1,x1,y2,x2 = faceLoc
            # y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4 
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED) #starting point on height reduced by -35 to be a little lower so we can write the name on top of this rectangle
            cv2.putText(img,name, (x2, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
        else:
            name = "unknown" 
            y1,x1,y2,x2 = faceLoc
            # y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4 
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED) #starting point on height reduced by -35 to be a little lower so we can write the name on top of this rectangle
            cv2.putText(img, name, (x2, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
            
    
    # img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)  #reduce size by scale 
    # img = cv2.resize(img, (900,600))
    
    cv2.imshow('Results',img)
    cv2.waitKey(1)
    
    
     