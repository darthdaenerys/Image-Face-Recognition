import cv2
import os
import numpy as np
import face_recognition as FR
import pickle

traindir='demoImages-master\known'

knownpersons=[]
knownencodings=[]

print('Training process has been started...')
for root,folders,files in os.walk(traindir):
    for file in files:
        imagepath=os.path.join(root,file)
        face=FR.load_image_file(imagepath)
        faceloc=FR.face_locations(face)[0]
        faceencode=FR.face_encodings(face)[0]
        name=os.path.splitext(file)[0]
        knownencodings.append(faceencode)
        knownpersons.append(name)

with open('encode_data.pkl','wb') as f:
    pickle.dump(knownpersons,f)
    pickle.dump(knownencodings,f)
    f.close()

print('Succesfully completed training')