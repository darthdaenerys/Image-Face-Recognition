import cv2
import os
import numpy as np
import face_recognition as FR
import pickle

testdir='demoImages-master\\unknown'
myfont=cv2.FONT_HERSHEY_PLAIN

def rescaleFrame(frame,scale=0.8):
    newframe=frame
    while(newframe.shape[1]>1280 or newframe.shape[0]>720):    
        width=int(newframe.shape[1]*scale)
        height=int(newframe.shape[0]*scale)
        newframe=cv2.resize(newframe,(width,height),interpolation=cv2.INTER_AREA)
    return newframe

with open('encode_data.pkl','rb') as f:
    knownpersons=pickle.load(f)
    knownencodings=pickle.load(f)
    f.close()

for root,folders,files in os.walk(testdir):
    for file in files:
        imgpath=os.path.join(root,file)
        img=FR.load_image_file(imgpath)
        facelocs=FR.face_locations(img)
        faceencodes=FR.face_encodings(img)

        color=(0,0,255)
        name='Unknown'
        size=2
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for faceloc,faceencode in zip(facelocs,faceencodes):
            matches=FR.compare_faces(knownencodings,faceencode)
            for idx,stat in enumerate(matches):
                if stat:
                    name=knownpersons[idx]
                    color=(0,255,0)
                    break
            top,right,bottom,left=faceloc
            cv2.rectangle(img,(left,top),(right,bottom),color,2)
            cv2.rectangle(img,(left,bottom),(right,bottom+30),color,-1)
            if right-left<150:
                size=1
            cv2.putText(img,name,(left,bottom+25),myfont,size,(0,0,0),2)
        img=rescaleFrame(img)
        cv2.imshow('Facial Recognition',img)
        cv2.moveWindow('Facial Recognition',0,0)
        cv2.waitKey(1000)