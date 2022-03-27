import cv2 
import numpy as np   
from pyzbar.pyzbar import decode  



cap = cv2.imread('card2.png')

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
            print('Authorized')
        else:
            print('Un-Authorized')
        #get the polygon points from the decoder
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1,1,2))
        #draw polygon around qr code
        cv2.polylines(cap, [pts], True, (0,255,0), 5)
        pts2 = barcode.rect
        cv2.putText(cap, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_COMPLEX, 2.5, (255,0,255), 2)
    
    h, w = cap.shape[:2]
    h = int(h/5)
    w = int(w/3)
    img = cv2.resize(cap, (h,w))
    cv2.imshow('Result', img)
    cv2.waitKey(1)
