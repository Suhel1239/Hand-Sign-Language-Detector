import cv2
# from cvzone.HandTrackingModule import HandDetector
import numpy as np
from PIL import Image
import math
import sklearn
import tensorflow as tf
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout, BatchNormalization
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# // *********** Hand Detector 
"""
Hand Tracking Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import math

import cv2
import mediapipe as mp


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):

        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param modelComplexity: Complexity of the hand landmark model: 0 or 1.
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                # if flipType:
                #     if handType.classification[0].label == "Right":
                #         myHand["type"] = "Left"
                #     else:
                #         myHand["type"] = "Right"
                # else:
                #     myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                if draw:
                    # self.mpDraw.draw_landmarks(img, handLms,
                    #                            self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    # cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                    #             2, (255, 0, 255), 2)

        return allHands, img
    
# // ****************************
    
enc = pickle.load(open('encoder.pkl','rb'))
dict ={1:"A" ,2:"B",3:"C",4:"D",5:"E",6:"F",7:"G",8:"H",9:"I",10:"K",11:"L",12:"M",13:"N",14:"O",15:"P"
       ,16:"Q",17:"R",18:"S",19:"T",20:"U",21:"V",22:"W",23:"X",24:"Y"}
offset = 15
imgsize = 280
cap = cv2.VideoCapture(0)
cap.set(3,800) 
cap.set(4,600)
cap.set(cv2.CAP_PROP_FPS, 30)
detector = HandDetector(maxHands=1)
text = "The Detected Text is : "
while (True):
    success, img = cap.read()
    hands, img = detector.findHands(img , draw=True)
    imgwhite = np.ones((imgsize,imgsize,3) , np.uint8)*255

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        lmList1 = []
        imgcrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgcropshape = imgcrop.shape
        aspectr = h/w 
        if aspectr>1:
            k=imgsize/h
            wcal = math.ceil(k*w)
            dim = (wcal , imgsize)
            imgresize = cv2.resize(imgcrop ,dim)
            imgresizeshape = imgresize.shape
            wgap = math.ceil((imgsize-wcal)/2)
            imgwhite[: , wgap:wcal+wgap] = imgresize
        else :
            k=imgsize/w
            hcal = math.ceil(k*h)
            dim = (imgsize , hcal)
            imgresize = cv2.resize(imgcrop ,dim)
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize-hcal)/2)
            imgwhite[hgap:hcal+hgap , :] = imgresize

    img_pil = Image.fromarray(imgwhite)
    imgwhite = np.array(img_pil.resize((28, 28)))

    # model = pickle.load(open('sign_lang_reduced_3d_50.pkl','rb'))
    # // **********************
    def create_model():
        # model=Sequential()
        # model.add(keras.Input(shape = (128,128,3))),
        # model.add(Conv2D(128,kernel_size=(5,5),
        #          strides=1,padding='same',activation='relu'))
        # model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
        # model.add(Conv2D(64,kernel_size=(2,2),
        #         strides=1,activation='relu',padding='same'))
        # model.add(MaxPool2D((2,2),2,padding='same'))
        # model.add(Conv2D(32,kernel_size=(2,2),
        #         strides=1,activation='relu',padding='same'))
        # model.add(MaxPool2D((2,2),2,padding='same'))
        # model.add(Conv2D(16,kernel_size=(2,2),
        #         strides=1,activation='relu',padding='same'))
        # model.add(MaxPool2D((2,2),2,padding='same'))
        # model.add(Conv2D(8,kernel_size=(2,2),
        #         strides=1,activation='relu',padding='same'))
        # model.add(Flatten())
        # model.add(Dense(units=512,activation='relu'))
        # model.add(Dropout(rate=0.25))
        # model.add(Dense(units=5,activation='softmax'))
        # model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        model = Sequential()
        model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Flatten())
        model.add(Dense(units = 512 , activation = 'relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units = 24 , activation = 'softmax'))
        model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
        return model
    
    model = create_model()
    # model.load_weights("/Users/suhelkhan/Computer_Vision/039_model.h5")
    model.load_weights("/Users/suhelkhan/Computer_Vision/hand_sign/kaggle_dataset.weights.h5")
    # // ************************
    imgwhite = cv2.cvtColor(imgwhite, cv2.COLOR_RGB2GRAY) 
    imgwhite = np.expand_dims(imgwhite , axis=2
                              )
    test = np.asarray(imgwhite).reshape(1,28,28,1)
    pred = model.predict(test)
    pred = np.argmax(pred)

    org = (30, 40) 
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1
    color = (150, 100, 50) 
    thickness = 2

   
    if(pred>24):
        cv2.putText(img=img , text=text + "" , org=org ,fontScale=fontScale ,fontFace=font , color=color , thickness=thickness)
    else:
        cv2.putText(img=img , text=text + dict[pred+1] , org=org ,fontScale=fontScale ,fontFace=font , color=color , thickness=thickness)
    
    cv2.imshow("Image", img)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing...")
        break
    if k%256 == 32:
        # ESC pressed
        text = text + str(dict[pred+1])
    if k%256 == 115:
        text = text + " "
cap.release()

cv2.destroyAllWindows()
