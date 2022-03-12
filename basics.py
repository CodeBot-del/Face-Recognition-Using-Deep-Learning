import cv2 
import numpy as np
import face_recognition

imgJack = face_recognition.load_image_file('imagesBasic/jackiechan.jpg')
#convert to rgb  
imgJack = cv2.cvtColor(imgJack,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('imagesBasic/brucelee.jpg')
#convert to rgb  
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#detect the face  
faceLoc = face_recognition.face_locations(imgJack)[0]
encodeJack = face_recognition.face_encodings(imgJack)[0]
#draw box around face 
cv2.rectangle(imgJack,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255),2)  #the arguments are: the image, starting points, ending points, color and thickness

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
#draw box around face 
cv2.rectangle(imgTest,(faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255),2) 

results = face_recognition.compare_faces([encodeJack], encodeTest)
#check how close are faces similar 
faceDis = face_recognition.face_distance([encodeJack], encodeTest)  #the lower the distance, the more similar the faces  

print(results, faceDis)
#write results on images 
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Jackie Chan', imgJack)
cv2.imshow('Jackie Test', imgTest)
cv2.waitKey(0)