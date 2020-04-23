import cv2, numpy as np, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from dataset import ShapeDataset
import pdb

img_size = 60
def flatten(dimData, images):
    images = np.array(images)
    images = images.reshape(len(images), dimData)
    images = images.astype('float32')
    images /=255
    return images
#read from existing data or generate data
if(not (os.path.exists('train_data.npz'))):
    folders, labels, images = ['triangle', 'star', 'square', 'circle'], [], []
    for folder in folders:
        print (folder)
        for path in os.listdir(os.getcwd()+'\\shapes\\'+folder):
            img = cv2.imread(os.getcwd()+'\\shapes\\'+folder+'\\'+path,0)
            #cv2.imshow('img', img)
            #cv2.waitKey(1)
            images.append(cv2.resize(img, (img_size, img_size)))
            labels.append(folders.index(folder))
        
    #break data into training and test sets
    to_train= 0
    train_images, test_images, train_labels, test_labels = [],[],[],[]
    for image, label in zip(images, labels):
        if to_train<5:
            train_images.append(image)
            train_labels.append(label)
            to_train+=1
        else:
            test_images.append(image)
            test_labels.append(label)
            to_train = 0
    '''
    idx = np.random.permutation(len(train_images))
    train_images,train_labels = np.array(train_images)[idx], np.array(train_labels)[idx]
    idx = np.random.permutation(len(test_images))
    test_images,test_labels = np.array(test_images)[idx], np.array(test_labels)[idx]
    '''
    np.savez('train_data.npz', images=train_images, labels=train_labels)
    np.savez('test_data.npz', images=test_images, labels=test_labels)

train_dataset=ShapeDataset('train_data.npz')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                          shuffle=True, num_workers=0)

test_dataset=ShapeDataset('test_data.npz')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16,
                                         shuffle=False, num_workers=0)
#train_images=train_dataset['images']
#train_labels=train_dataset['labels']
#test_images=test_dataset['images']
#test_labels=test_dataset['labels']
#classes = ('triangle', 'star', 'square', 'circle')





#CNN network defination
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
def adjust_learning_rate(epoch):

    lr = 0.001

    if epoch > 150:
        lr = lr / 1000
    elif epoch > 100:
        lr = lr / 100
    elif epoch > 50:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

cnn = CNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
cnn.to(device)
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)
print('On gpu :',next(cnn.parameters()).is_cuda)

nepo=10
for epoch in range(nepo):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)
        labels = labels.squeeze(1).long()
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('epoch [',epoch+1,'/',nepo,'] loss :',running_loss/len(train_loader))
    adjust_learning_rate(epoch)
print('Finished Training')
PATH = './shape_net.pth'
torch.save(cnn.state_dict(), PATH)
'''
PATH = './shape_net.pth'
cnn.load_state_dict(torch.load(PATH))
#test
correct = 0
total = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)
        labels = labels.squeeze(1).long()
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        correct += (predicted == labels).sum().item()
        #pdb.set_trace()
print(len(train_loader))

print('Accuracy of the network on the 3119 test images: %f %%' % (
    100 * correct / total))