import numpy as np
import os
from matplotlib import pyplot as plt


class SIR():
    def __init__(self):
        self.H = 64
        self.W = 64
        self.all = self.H * self.W
        # n means the total counts of simulation
        self.n = 1000

        # 0 S green
        # 1 I red
        # 2 R blue
        self.map = np.zeros((self.H,self.W),dtype='uint8')
        self.map2 = np.zeros((self.H,self.W),dtype='uint8')
        # p1 means S to I
        self.p1 = 0.95
        # p2 means I to R
        self.p2 = 0.30
        # output img RGB
        self.img = np.zeros((self.H,self.W,3),dtype='uint8') 
        self.img[:,:,1] = 255
        self.num_init_I = 10
        self.num_init_R = 0
        
        count = 0
        while count < self.num_init_I:
            i = np.random.randint(0,self.H)
            j = np.random.randint(0,self.W)
            if self.map[i,j] != 0:
                continue
            self.map[i,j] = 1
            count += 1

        count = 0
        while count < self.num_init_R:
            i = np.random.randint(0,self.H)
            j = np.random.randint(0,self.W)
            if self.map[i,j] != 0:
                continue
            self.map[i,j] = 2
            count += 1

        self.map2 = self.map.copy()

        for i in range(self.H):
            for j in range(self.W):
                if self.map[i,j] == 1:
                    self.img[i,j,0] = 255
                    self.img[i,j,1] = 0
                    self.img[i,j,2] = 0
                elif self.map[i,j] == 2:
                    self.img[i,j,0] = 0
                    self.img[i,j,1] = 0
                    self.img[i,j,2] = 255
                    

        # self.img_path = './SIR_img/'
        # if not os.path.isdir(self.img_path):
        #     os.makedirs(self.img_path)



    def set_I(self,i,j):
        if i < 0 or i > self.H - 1:
            return
        if j < 0 or j > self.W - 1:
            return
        if self.map[i,j] == 0 and np.random.rand() < self.p1:
            self.map2[i,j] = 1
            self.img[i,j,0] = 255
            self.img[i,j,1] = 0
            self.img[i,j,2] = 0

    def green(self,i,j):
        if i < 0 or i > self.H - 1 or j < 0 or j > self.W - 1:
            return
        self.map2[i,j] = 0
        self.img[i,j,0] = 0
        self.img[i,j,1] = 255
        self.img[i,j,2] = 0

    def red(self,i,j):
        if i < 0 or i > self.H - 1 or j < 0 or j > self.W - 1:
            return
        self.map2[i,j] = 1
        self.img[i,j,0] = 255
        self.img[i,j,1] = 0
        self.img[i,j,2] = 0

    def blue(self,i,j):
        if i < 0 or i > self.H - 1 or j < 0 or j > self.W - 1:
            return
        self.map2[i,j] = 2
        self.img[i,j,0] = 0
        self.img[i,j,1] = 0
        self.img[i,j,2] = 255


    # def set_S(self,i,j):
    #     if i < 0 or i > self.H - 1:
    #         return
    #     if j < 0 or j > self.W - 1:
    #         return
    #     self.map2[i,j] = 0
    #     self.img[i,j,0] = 0
    #     self.img[i,j,1] = 255
    #     self.img[i,j,2] = 0

    def set_R(self,i,j):
        if i < 0 or i > self.H - 1:
            return
        if j < 0 or j > self.W - 1:
            return
        if self.map[i,j] == 1 and np.random.rand() < self.p2:
            self.map2[i,j] = 2
            self.img[i,j,0] = 0
            self.img[i,j,1] = 0
            self.img[i,j,2] = 255
            return True

    def simulate(self):
        dx = [0,0,1,-1]
        dy = [1,-1,0,0]

        plt.ion()

        for epoch in range(self.n):
            for x in range(self.H):
                for y in range(self.W):
                    if self.map[x,y] == 0:
                        continue
                    elif self.map[x,y] == 1:
                        if not self.set_R(x,y):
                            for x1,y1 in zip(dx,dy):
                                self.set_I(x+x1,y+y1)
                    else:
                        continue
            self.map = self.map2.copy()
            
            plt.cla()
            plt.imshow(self.img)
            # plt.text(self.H-1,0,f'n = {epoch}',fontsize=18)
            plt.title(f'n = {epoch}',fontsize=18)
            plt.pause(0.1)
            
            S_percent = np.where(self.map==0,1,0).mean()
            I_percent = np.where(self.map==1,1,0).mean()
            R_percent = np.where(self.map==2,1,0).mean()


            print(f'{epoch}/{self.n} , ' + f'S = {S_percent} , I = {I_percent} , R = {R_percent}')
            if np.where(self.map==1,1,0).sum() == 0:
                print('exit...')
                break
        plt.ioff()
        plt.show()



class SIR2():
    def __init__(self):
        super().__init__()
        self.pause_time = 0.3
        self.H = 64
        self.W = 64
        self.n = 100
        # file mode use p1
        self.p1 = 1e-3
        self.p2 = 0.5
        self.img = np.zeros((self.H,self.W,3),dtype='uint8')
        self.img[:,:,1] = 255
        self.map = np.zeros((self.H,self.W))
        # 'file' or 'user'
        self.mode = 'file'
        # user mode use Gaussian distribution
        self.mean = 1e-3
        self.std = (1e-2 - 1e-8) / 6
        self.epsilon = 1e-12
        self.pmap = np.clip(np.random.randn(self.H,self.W) * self.std + self.mean,self.epsilon,1)
        # S 0 green
        # I 1 red
        # R 2 blue

        self.num_init_I = 1
        count = 0
        while count < self.num_init_I:
            i = np.random.randint(0,self.H)
            j = np.random.randint(0,self.W)
            if self.map[i,j] != 0:
                continue
            self.map[i,j] = 1
            self.img[i,j,0] = 255
            self.img[i,j,1] = 0
            self.img[i,j,2] = 0
            count += 1
        
        self.map2 = self.map.copy()

    @property
    def num_S(self):
        return np.where(self.map==0,1,0).sum()
    @property
    def num_I(self):
        return np.where(self.map==1,1,0).sum()
    @property
    def num_R(self):
        return np.where(self.map==2,1,0).sum()
    
    def set_S(self):
        pass


    def set_I(self):
        p = (1 - self.p1) ** self.num_I
        points = np.where(self.map==0)
        for x,y in zip(*(points)):
            # p = 1
            # for _ in range(self.num_I):
            #     p *= (1 - np.clip(np.random.randn() * self.std + self.mean,self.epsilon,1))
            if np.random.rand() < p:
                continue
            else:
                self.map2[x,y] = 1
                self.img[x,y,0] = 255
                self.img[x,y,1] = 0
                self.img[x,y,2] = 0
                
        self.map = self.map2.copy()


    def set_R(self):
        points = np.where(self.map==1)
        for x,y in zip(*(points)):
            if np.random.rand() >= self.p2:
                continue
            else:
                self.map2[x,y] = 2
                self.img[x,y,0] = 0
                self.img[x,y,1] = 0
                self.img[x,y,2] = 255


    def simulate(self):
        plt.ion()
        for epoch in range(self.n):
            self.set_I()
            plt.cla()
            plt.imshow(self.img)
            plt.title(f'epoch = {epoch}')
            plt.pause(self.pause_time)
            self.set_R()
            plt.cla()
            plt.imshow(self.img)
            plt.title(f'epoch = {epoch}')
            plt.pause(self.pause_time)
            print(f'{epoch}/{self.n}')
            if self.num_I == 0:
                break
        print('exit...')
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    sir = SIR()
    sir.simulate()
    