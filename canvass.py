import numpy as np
from PIL import Image



class canvas:
    def __init__(self):
        self.windowsize=64
        self.canvas=np.ones((self.windowsize,self.windowsize),np.uint8)*255
        self.action_space=8
        self.similarity=0.8
        self.state_space=self.windowsize*self.windowsize*2

        self.brushx=self.windowsize//2
        self.brushy=self.windowsize//2
        self.reset()
        self.showimg()

    def reset(self):
        self.canvas=np.ones((self.windowsize,self.windowsize),np.uint8)*255
        self.brushx=self.windowsize//2
        self.brushy=self.windowsize//2

    def showimg(self):
        im=Image.fromarray(np.uint8(self.canvas))
        im.show()

    # decide horizon(whether the pictures match)
    def get_similarity(self,cnn_output):
        done = bool(cnn_output >= self.similarity)
        return done

    def step(self,action):
        # 0 1 2
        # 3 P 4
        # 5 6 7
        if action==0:
            self.brushx -= 1
            self.brushy -= 1
        elif action==1:
            self.brushy -= 1
        elif action==2:
            self.brushx += 1
            self.brushy -= 1
        elif action==3:
            self.brushx -= 1
        elif action==4:
            self.brushx += 1
        elif action==5:
            self.brushx -= 1
            self.brushy += 1
        elif action==6:
            self.brushy += 1
        elif action==7:
            self.brushx += 1
            self.brushy += 1
        # if the brush hits the frame, set the brush to the center
        if (self.brushx<0 or self.brushx>31 or self.brushy<0 or self.brushy>31):
            self.brushx = self.windowsize // 2
            self.brushy = self.windowsize // 2
            # self.brushx = np.random.randint(0, self.windowsize)
            # self.brushy = np.random.randint(0, self.windowsize)
        self.canvas[self.brushy,self.brushx]=0

    def get_reward(self,done):
        if done:
            reward=1000.0
        else:
            reward=-1.0
        return reward














