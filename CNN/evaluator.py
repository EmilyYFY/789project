# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:58:17 2020

@author: zhong
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
import skimage.color
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(1, 64, 3,padding =1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(64, 64, 3,padding =1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(0.3),
            nn.Conv2d(64, 128, 3,padding =1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Conv2d(128, 128, 3,padding =1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(0.4),
            nn.Conv2d(128, 256, 3,padding =1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Conv2d(256, 256, 3,padding =1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3,padding =1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Conv2d(512, 512, 3,padding =1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(0.4),
            nn.Conv2d(512, 512, 3,padding =1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Conv2d(512, 512, 3,padding =1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.ly=nn.Sequential(nn.Linear(512,512),
            #nn.Dropout(0.5),
            nn.Linear(512,4))
            

    def forward(self, x):
        x = self.convlayer(x)
        x = x.view(-1, 512)
        #x = F.softmax(self.fc(x),dim=1)
        x = self.ly(x)
        #print(x)
        return x
PATH='./shape_net.pth'
net=CNN().double()
net.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
def evaluate(data,label):
    image=np.zeros((1,1,60,60))
    image[0,0,:,:]=data
    image=torch.tensor(image)
    output = net(image)
    x = F.softmax(output,dim=1)
    x=x.data.numpy()
    prob=x[0,label]
    threshold=0.995 #probability where reward change from negative to positive
    negreward=-2    #negtive reward when prob is small
    maxreward=100   #reward when probability =1
    a=(np.log(maxreward-negreward)-np.log(-negreward))/(1-threshold)
    b=a*threshold-np.log(-negreward)
    reward=np.exp(a*prob-b)+negreward
    return reward
import matplotlib.pyplot as plt
if __name__ == "__main__":
    for i in range(16):
        path="./test_doodle/"+str(i+1)+".jpg"
        #print(path)
        img = Image.open( path )
        data = np.asarray(img)
        if data.shape[-1]==3:
            data=skimage.color.rgb2gray(data)
        
        #print(data.shape)
        fig, ax = plt.subplots()
        fig.set_size_inches(1, 1)
        ax.imshow(img,cmap='Greys_r')
        plt.show()
        #evaluate(data,label) here label from 0~3
        print("Triangle: {:.8f} Star: {:.8f} Square: {:.8f} Circle: {:.8f}".format(evaluate(data,0),evaluate(data,1),evaluate(data,2),evaluate(data,3)))