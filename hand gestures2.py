import cv2
import time
import mediapipe as mp
import pyautogui #keyboard
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from keras.layers import Dropout, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D

import pickle

imageDimensions = (32, 32)
noOfClasses = 7


def count_fingers(list):# result.multi_handlandmarks  (or)hands
    cnt = 0

    thresh = (list.landmark[0].y * 100 - list.landmark[9].y * 100) / 2#23,24
    print(thresh)
    if (list.landmark[5].y * 100 - list.landmark[8].y * 100) > thresh:
        cnt += 1#1
    if (list.landmark[9].y * 100 - list.landmark[12].y * 100) > thresh:
        cnt += 1#1+1=2
    if (list.landmark[13].y * 100 - list.landmark[16].y * 100) > thresh:
        cnt += 1#2+1=3
    if (list.landmark[17].y * 100 - list.landmark[20].y * 100) > thresh:
        cnt += 1#3+1=4
    if (list.landmark[5].x * 100 - list.landmark[4].x * 100) > thresh:
        cnt += 1#4+1=5
    return cnt


x1 = y1 = x2 = y2 = 0
cap = cv2.VideoCapture(0)
drawing = mp.solutions.drawing_utils
handss = mp.solutions.hands
hand_obj = handss.Hands(max_num_hands=1)



# def myModel():
#     noOfFilters = 60  # image size
#     sizeOfFilter1 = (5, 5)  # kernel size
#     sizeOfFilter2 = (3, 3)  # kernel_size
#     sizeOfPool = (2, 2)  # max pooling
#     noOfNodes = 500  # input size neuron imagee size  vachi neuron edupom so 32*32
#
#     model = Sequential()
#     model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],  # 60
#                                                                imageDimensions[1], 1), activation='relu')))
#     model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))  # 60
#     model.add(MaxPooling2D(pool_size=sizeOfPool))
#
#     model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))  # 30
#     model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))  # 30
#     model.add(MaxPooling2D(pool_size=sizeOfPool))
#
#     model.add(MaxPooling2D(pool_size=sizeOfPool))
#     model.add(Dropout(0.5))
#
#     model.add(Flatten())
#     model.add(Dense(noOfNodes, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(noOfClasses, activation='softmax'))
#
#     model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#     # print("mp",model.predict())
#     return model
#

while True:

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    res = hand_obj.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        hand_keypoints = res.multi_hand_landmarks[0] #hands irruku
        landmarks = hand_keypoints.landmark#fingers landmark  x and y axis

        #.............................................................................
        cnt = count_fingers(hand_keypoints)
                                        #hand landmarks x and y axis     #hands image train cnn
        drawing.draw_landmarks(frame, res.multi_hand_landmarks[0],     handss.HAND_CONNECTIONS)


        if (cnt == 1):
            pyautogui.press("right")

        elif (cnt == 2):
             pyautogui.press("left")
        elif (cnt == 3):
            pyautogui.press("up")

        elif (cnt == 4):
            pyautogui.press("down")
        elif (cnt == 5):

            pyautogui.scroll(10)
            print("S")
    cv2.imshow("wimdow", frame)
    cv2.waitKey(1)