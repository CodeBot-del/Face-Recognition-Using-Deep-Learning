import cv2 
import numpy as np 
import face_recognition 
import os
from addface import encodedFaces

# file = open("encodings.txt", "r")
# content = file.read()
# encodingsKnown = np.array(content)
# file.close()
encodingsKnown = encodedFaces

# print(encodingsKnown)

#get the match to perform recognition
cap = cv2.imread('./imagesTest/assembly.jpg')

while True:
    # success, img = cap.read() #this line to be used when reading from webcam
    img = cap 
    imgS = cv2.resize(img,(0,0), None,0.25,0.25) #compress the image to enhance performance
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS) #find the locations of all faces in the image
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) #find encodings of facees in the image
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): #this grabs both the encodings and the locations of the faces in the current frame
        matches = face_recognition.compare_faces(encodingsKnown, encodeFace) #compare the image encodings with the known encodings
        faceDis = face_recognition.face_distance(encodingsKnown, encodeFace) #Find how much the faces differ between the image and the known
        
        matchIndex = np.argmin(faceDis) #take the lowest face distance to be the best match
        if matchIndex < 0.12:            #accept match of lower than 0.1, for the sake of accuracy
            matchIndex = matchIndex
            
        #draw green boxes around verified faces 
        if matches[matchIndex]:
            name = "Authorized"
            y1, x1, y2, x2 = faceLoc
            # y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4 
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x2, y2-6), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            
        else:
            name = "Unknown"
            y1, x1, y2, x2 = faceLoc
            # y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4 
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name, (x2, y2-6), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            
    
    cv2.imshow('Results', img)
    cv2.waitKey(1)
            
            
        