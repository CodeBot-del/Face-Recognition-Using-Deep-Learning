
#GUYS CHAT YANGU INASUMBUA.... NTAKUA NAANDIKIA APA KWA COMMENTS

#SO... SYSTEM INAPITISHA STEPS TATU 
#1. LOAD THE PHOTOS
#2. FACE LOCATION DETECTION
#3. ENCODE KWA KUTUMIA ALGORITHM YA HOG. (128 ARRAYS YA VALUES)

#RECOGNITION 
#LOAD NEW IMAGE 
#LOCATION YA FACE 
#ENCODE ILE SURA
#COMPARE ENCODINGS MPYA NA ZILE ZILIZOPO & check distance (check for the lowest distnce)


#ISSUE IPO KENE PERFORMANCE
#KILA MDA TUKI RUN, LAZIMA ITA ENCODE SURA ZOTE ZILIZOPO, THEN ITA ENCODE SURA MPYA, HALAFU ITATOA MAJIBU. 
#WHAT IF TUNA PICHA ELFU MOJA... ITACHUKUA MASAA KU RUN PROGRAM. 
#SOLUTION
#TUSIWE TUNA ENCODE KILA MUDA TUKI RUN... BALI TUKI RUN TUWE TUNA ENCODE SURA MPYA TU, AMBAYO INACHUKUA MILLISECONDS KU ENCODE
#TUNA COMPARE NA ENCODINGS ZILIZOPO, AMBAPO ZILE ENCODINGS ZIWE SAVED.
#TUKISHA ENCODE SURA ZOTE, TU SAVE KENE CSV FILE. TUTAKUA TUNA ILOAD IYO FILE YENYE ENCODINGS
#WHICH MEANS HATUTA HITAJI KU ENCODE KILA MDA KAMA TUKI SAVE ZILE ENCODINGS. 

#KO HAPO TUTAKUA NA SUBSYSTEMS MBILI.... MOJA YA KU ENCODE TU, MOJA YA KU COMPARE... AU KU RECOGNIZE. 
#ILE YA ENCODING ITAKUA INA SAVE KENE CSV... ALAF YA RECOGNITION INA ENCODE SURA MPYA, INA COMPARE APOAPO. 

#KO KAMA KUNA WATU 1000... ITABIDI WAWE ENCODED MARA MOJA TU, THEN ZILE CODE ZITAPITISHWA KENE SYSTEM YA RECOGNITION

#HII ITA IMPROVE PERFORMANCE, ITAPUNGUZA DELAY TIME.... SYSTEM ITAKUA NYEPESI.

#KO HII ITABIDI IWE SPLITTED KENE SYSTEM MBILI... AMBAPO HII NTAIFANYA KUANZIA MONDAY.





#import the libraries
import cv2 
import numpy as np #for maths
import face_recognition #inakuja na functions za ku encode, compare, kucheck distance....
import os   #kwa ajili ya kuread filesystem

path = 'imagesRecognition' #hapa nimevuta file la picha... linaitwa ivo imagesRecognition...xawa nimekupt master
images = []
classNames = []
myList = os.listdir(path)
print(myList)

#import the images from the directory.... tunazi import picha zote
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
        encode = face_recognition.face_encodings(img)[0] #tuna encode zile face zilizopo
        encodeList.append(encode)
    return encodeList
                                         #master sorry lkn mbona cjaona ulipovuta file la image....
encodeListKnown = findEncodings(images)  #apo sasa tuna encode kutumia iyo function yume declare
print('################------ Encoding Complete ------################') #tuna print iyo ili kujua encoding ikishaisha
print(encodeListKnown) ##HHAPA NDO INAPRINT IZO ARRAY, ZIPO 128 BOSS..

##HAPA SASA TUMEMALIZA KU ENCODE SURA ZINAZOJULIKANA

#get the image to match with... For now we read an image, later on we will read from webcam (cv2.VideoCapture())
#KWA SAIVI, TUNA READ IMAGE, BAADAE TUTAWEKA WEBCAM
cap = cv2.imread('2.jpg')  #apa tuna read image moja kwa moja cuz amna webcam

while True:
    # success, img = cap.read() #this line to be used when reading from webcam
    img = cap  #tume assign kene variable img ile picha yetu MPYA
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) #compress the image to improve performance
    imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #apa tunaibadilisha picha kuwa grayscale
    
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
            
            #uwe unacommentia huku pia ili tukuelewe master
            name = classNames[matchIndex].upper()  #Get the Name of the image(person) that matched successfully
            # print(name)
            y1,x1,y2,x2 = faceLoc
            # y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4 
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED) #starting point on height reduced by -35 to be a little lower so we can write the name on top of this rectangle
            cv2.putText(img,name, (x2, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
        else:
            name = "unknown" #kama haja match
            y1,x1,y2,x2 = faceLoc
            # y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4 
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED) #starting point on height reduced by -35 to be a little lower so we can write the name on top of this rectangle
            cv2.putText(img,name, (x2, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            
            
    
    # img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)  #reduce size by scale 
    # img = cv2.resize(img, (900,600))
    cv2.imshow('Results',img)
    cv2.waitKey
    
    
    