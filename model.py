import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR

class DeepDualNet(nn.Module):

    def __init__(self):
        super(DeepDualNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)      # 64 x 64
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)    # 32 x 32
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)   # 16 x 16
        self.conv4 = nn.Conv2d(256, 384, 3, 1, 1)   # 16 x 16
        self.conv5 = nn.Conv2d(384, 512, 4, 2, 1)   # 8 x 8
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, dilation=4, padding=4)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, dilation=8, padding=8)
        self.conv9 = nn.Conv2d(512, 256, 3, 1, 1)
        self.IN_64 = nn.InstanceNorm2d(64)
        self.IN_128 = nn.InstanceNorm2d(128)
        self.IN_256 = nn.InstanceNorm2d(256)
        self.IN_384 = nn.InstanceNorm2d(384)
        self.IN_512 = nn.InstanceNorm2d(512)

        self.fullyC_percep = nn.Sequential(
                nn.Linear(8192, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 2))
        
        self.feature_xz = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3,1,1),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True))
        
        self.fullyC_xz = nn.Sequential(
            nn.Linear(8192, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 2))
        
        self.feature_per1 = nn.Sequential(
            nn.Conv2d(64,64,4,2,1), #32 x 32
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,4,2,1), # 16 x 16
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,4,2,1), # 8 x 8
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        
        self.feature_per2 = nn.Sequential(
            nn.Conv2d(128,64,4,2,1), #16 x 16
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,4,2,1), #8 x 8
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        
        self.feature_per3 = nn.Sequential(
            nn.Conv2d(256,128,4,2,1), #8 x 8
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,1,1),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        
        self.feature_per4 = nn.Sequential(
            nn.Conv2d(384,256,4,2,1), #8 x 8
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,1,1),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,1,1), 
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        
        self.feature_per5 = nn.Sequential(
            nn.Conv2d(512,256,3,1,1), #8 x 8
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,1,1),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,1,1),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        
        self.feature_per6 = nn.Sequential(
            nn.Conv2d(512,256,3,1,1), #8 x 8
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,1,1),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,1,1), 
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        
        self.feature_per7 = nn.Sequential(
            nn.Conv2d(512,256,3,1,1), #8 x 8
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,1,1),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,1,1),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        
        self.feature_per8 = nn.Sequential(
            nn.Conv2d(512,256,3,1,1), #8 x 8
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,1,1),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,1,1),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True))

        self.feature_per9 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1), #8 x 8
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,1,1),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True))

        self.feature_per = nn.Sequential(
            nn.Conv2d(576,512,3,1,1), #8 x 8
            nn.InstanceNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,1,1),
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,1,1),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True))
            
        # initailize
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.xavier_normal_(self.conv7.weight)
        nn.init.xavier_normal_(self.conv8.weight)
        nn.init.xavier_normal_(self.conv9.weight)

    def extract_feature(self, x):
        x1 = F.relu(self.IN_64(self.conv1(x)))
        x2 = F.relu(self.IN_128(self.conv2(x1)))
        x3 = F.relu(self.IN_256(self.conv3(x2)))
        x4 = F.relu(self.IN_384(self.conv4(x3)))
        x5 = F.relu(self.IN_512(self.conv5(x4)))
        x6 = F.relu(self.IN_512(self.conv6(x5)))
        x7 = F.relu(self.IN_512(self.conv7(x6)))
        x8 = F.relu(self.IN_512(self.conv8(x7)))
        x9 = F.relu(self.IN_256(self.conv9(x8)))
        return x1,x2,x3,x4,x5,x6,x7,x8,x9

    def forward(self, x,z):
        # weight sharing
        x1,x2,x3,x4,x5,x6,x7,x8,x9 = self.extract_feature(x)
        z1,z2,z3,z4,z5,z6,z7,z8,z9 = self.extract_feature(z)

        # calculate the diff of left and right feature maps
        per1=abs(x1-z1)
        per2=abs(x2-z2)
        per3=abs(x3-z3)
        per4=abs(x4-z4)
        per5=abs(x5-z5)
        per6=abs(x6-z6)
        per7=abs(x7-z7)
        per8=abs(x8-z8)
        per9=abs(x9-z9)

        # encoding all diff of feature maps to 8 x 8
        per1=self.feature_per1(per1)
        per2=self.feature_per2(per2)
        per3=self.feature_per3(per3)
        per4=self.feature_per4(per4)
        per5=self.feature_per5(per5)
        per6=self.feature_per6(per6)
        per7=self.feature_per7(per7)
        per8=self.feature_per8(per8)
        per9=self.feature_per9(per9)

        # concatenate all feature maps
        per_cat=torch.cat([per1,per2,per3,per4,per5,per6,per7,per8,per9],dim=1)
        n, c, h, w = x.size()

        # concatenate last feature maps of left and right image for fully connected layer
        xz=torch.cat([x9,z9],dim=1)
        xz=self.feature_xz(xz)

        # fully connected layers of xz and diff
        out=xz.view(n,-1)
        out= self.fullyC_xz(out)
        out2=self.feature_per(per_cat)
        out2=out2.view(n,-1)
        out2= self.fullyC_percep(out2)

        return out, out2

class DeepDual():

    def __init__(self, net_path=None, **kargs):
        super(DeepDual, self).__init__
        self.cfg = self.parse_args(**kargs)
        self.CEloss =nn.CrossEntropyLoss()
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda')
        self.net = DeepDualNet()
        """
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        """
        #self.net = nn.DataParallel(self.net,device_ids=[0,1]).cuda()
        self.net = nn.DataParallel(self.net).cuda()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        self.lr_scheduler = ExponentialLR(
            self.optimizer, gamma=self.cfg.lr_decay)

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            # train parameters
            'initial_lr': 0.01,
            'lr_decay': 0.8685113737513527,
            'weight_decay': 5e-4,
            'momentum': 0.9}

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def step(self, left_img,right_img,labels, backward=True, update_lr=True):
        if backward:
            self.net.train()
            if update_lr:
                self.lr_scheduler.step()
        else:
            self.net.eval()

        z = left_img.to(self.device)
        x = right_img.to(self.device)
        with torch.set_grad_enabled(backward):
            responses,responses2 = self.net(x,z)
            # for test
            if backward==0:
                test1=responses.view(-1).data[0]
                test2=responses2.view(-1).data[0]
                test=test1+test2
                return test.item()
            # for training
            if backward:
                labels=labels.view(-1).long()
                loss1 = self.CEloss(
                    -responses, labels)
                loss2 = self.CEloss(
                    -responses2, labels)
                loss=loss1+loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Check the number of parameters
                #pytorch_total_params = sum(p.numel() for p in self.net.parameters())
                #print(pytorch_total_params)
        return loss.item()