# import torch
# from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())

# import cv2, numpy as np

# meas=[]
# pred=[]
# frame = np.zeros((400,400,3), np.uint8) # drawing canvas
# mp = np.array((2,1), np.float32) # measurement
# tp = np.zeros((2,1), np.float32) # tracked / prediction

# def onmouse(k,x,y,s,p):
#     global mp,meas
#     mp = np.array([[np.float32(x)],[np.float32(y)]])
#     meas.append((x,y))

# def paint():
#     global frame,meas,pred
#     for i in range(len(meas)-1): cv2.line(frame,meas[i],meas[i+1],(0,100,0))
#     for i in range(len(pred)-1): cv2.line(frame,pred[i],pred[i+1],(0,0,200))

# def reset():
#     global meas,pred,frame
#     meas=[]
#     pred=[]
#     frame = np.zeros((400,400,3), np.uint8)

# cv2.namedWindow("kalman")
# cv2.setMouseCallback("kalman",onmouse);
# kalman = cv2.KalmanFilter(4,2)
# kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
# kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
# kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
# #kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003
# while True:
#     kalman.correct(mp)
#     tp = kalman.predict()
#     pred.append((int(tp[0]),int(tp[1])))
#     paint()
#     cv2.imshow("kalman",frame)
#     k = cv2.waitKey(30) &0xFF
#     if k == 27: break
#     if k == 32: reset()

# t = ([1, 2], [23, 4], [3, 34], [3, 24], [23,655])
# t=np.array(t)
# print(t[:,:3])

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30])
iced_tea_sales = np.array([77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84])

lm = LinearRegression()
lm.fit(np.reshape(temperatures, (len(temperatures), 1)), np.reshape(iced_tea_sales, (len(iced_tea_sales), 1)))

# 新的氣溫
to_be_predicted = np.array([30])
predicted_sales = lm.predict(np.reshape(to_be_predicted, (len(to_be_predicted), 1)))

# 視覺化
plt.scatter(temperatures, iced_tea_sales, color='black')
plt.plot(temperatures, lm.predict(np.reshape(temperatures, (len(temperatures), 1))), color='blue', linewidth=3)
plt.plot(to_be_predicted, predicted_sales, color = 'red', marker = '^', markersize = 10)
plt.xticks(())
plt.yticks(())
plt.show()