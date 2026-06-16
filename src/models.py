import torch, torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, n=7):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(16*24*24, n))
    def forward(self,x): return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, n=7):
        super().__init__()
        def b(i,o): return [nn.Conv2d(i,o,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.f = nn.Sequential(*b(1,32), *b(32,64), *b(64,128), nn.Flatten())
        self.c = nn.Sequential(nn.Linear(128*6*6,256), nn.ReLU(), nn.Linear(256,n))
    def forward(self,x): return self.c(self.f(x))

class CNN_BN(nn.Module):
    def __init__(self, n=7, p=0.4):
        super().__init__()
        def b(i,o): return [nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
        self.f = nn.Sequential(*b(1,32), *b(32,64), *b(64,128), nn.Flatten())
        self.c = nn.Sequential(nn.Dropout(p), nn.Linear(128*6*6,256), nn.ReLU(), nn.Dropout(p), nn.Linear(256,n))
    def forward(self,x): return self.c(self.f(x))

class VGGStyle(nn.Module):
    def __init__(self, n=7, p=0.4):
        super().__init__()
        def b(i,o): return [nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                            nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
        self.f = nn.Sequential(*b(1,64), *b(64,128), *b(128,256), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.c = nn.Sequential(nn.Dropout(p), nn.Linear(256, n))
    def forward(self,x): return self.c(self.f(x))

class ResBlock(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.c1=nn.Conv2d(i,o,3,s,1,bias=False); self.b1=nn.BatchNorm2d(o)
        self.c2=nn.Conv2d(o,o,3,1,1,bias=False); self.b2=nn.BatchNorm2d(o)
        self.sc=nn.Sequential() if s==1 and i==o else nn.Sequential(nn.Conv2d(i,o,1,s,bias=False), nn.BatchNorm2d(o))
    def forward(self,x):
        y=torch.relu(self.b1(self.c1(x))); y=self.b2(self.c2(y))
        return torch.relu(y+self.sc(x))

class ResNetMini(nn.Module):
    def __init__(self, n=7, p=0.3):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(1,64,3,1,1,bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.l1=nn.Sequential(ResBlock(64,64),    ResBlock(64,64))
        self.l2=nn.Sequential(ResBlock(64,128,2), ResBlock(128,128))
        self.l3=nn.Sequential(ResBlock(128,256,2),ResBlock(256,256))
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p), nn.Linear(256,n))
    def forward(self,x): return self.head(self.l3(self.l2(self.l1(self.stem(x)))))

MODELS = {'tiny':TinyCNN, 'simple':SimpleCNN, 'cnn_bn':CNN_BN, 'vgg':VGGStyle, 'resnet':ResNetMini}
