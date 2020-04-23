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
    return x[0,label]
if __name__ == "__main__":
    for i in range(10):
        path="./test_doodle/"+str(i+1)+".jpg"
        print(path)
        img = Image.open( path )
        data = np.asarray(img)
        #print(data.shape)
        
        #evaluate(data,label) here label from 0~3
        print(evaluate(data,0))