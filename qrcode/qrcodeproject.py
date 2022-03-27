import cv2 
import numpy as np   
from pyzbar.pyzbar import decode  



cap = cv2.imread('fraud.png')

# cap.set(3,640)
# cap.set(4,480)

with open('data.txt') as f:
    Authenticated = f.read().splitlines()


while True:
    
    #in case of multiple barcodes
    for barcode in decode(cap):
        myData = barcode.data.decode('utf-8')
        print(myData)
        
        if myData in Authenticated:
            myOutput = 'Authorized'
            myColor = (0,255,0)
        else:
            myOutput = 'Un-Authorized'
            myColor = (0,0,255)
        #get the polygon points from the decoder
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1,1,2))
        #draw polygon around qr code
        cv2.polylines(cap, [pts], True, myColor, 5)
        pts2 = barcode.rect
        cv2.putText(cap, myOutput, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_COMPLEX, 2.5, myColor, 2)
    
    h, w = cap.shape[:2]
    h = int(h/5)
    w = int(w/3)
    img = cv2.resize(cap, (h,w))
    cv2.imshow('Result', img)
    cv2.waitKey(1)
