from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader
import loader_BC as loader
# from got10k.datasets import ImageNetVID, GOT10k
import glob
from pairwise import Pairwise
from siamfc_BC import TrackerSiamFC,Encoder
import random

from torch import nn
import numpy
import csv
# from binary_classification import BCNet

import torch.nn.functional as F


def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+numpy.exp(-x))

loss_path='./loss/lossBC0816.csv'
if not os.path.exists('./loss/'):
    os.makedirs('./loss/')
    f = open(loss_path, 'w', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(["Epoch","NETLoss"])
    f.close()

if __name__ == '__main__':
    # setup dataset
    dirname='../../DCVL_Face_DB/'
    # dirnames.extend(sorted(glob.glob('../../DCVL_Face_DB/')))    
    # dirnames=list(OrderedDict.fromkeys(dirnames))
    
    # setup data loader
    # setup tracker
    cuda = torch.cuda.is_available()
    #device = torch.device('cuda:0' if cuda else 'cpu')

    # path for saving checkpoints
    net_dir = 'model'
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)
    epoch_num = 30
    mask=list(range(1,36))
    #random.shuffle(mask)
    for trial in range(50):
        if mask[0]>0:
            tracker = TrackerSiamFC()
            net_path=os.path.join(net_dir,'model_%02d.pth'%(trial))
            
            if net_path is not None:
                tracker.net.load_state_dict(torch.load(
                    net_path, map_location=lambda storage, loc: storage))
                print('Loaded Model')

            responses=[]
            cl=[]
            rf=[]
            wbuffer=[]
            f = open('./test/similarity_BC0825%02d.csv'%(trial), 'a', encoding='utf-8')
            wr = csv.writer(f,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            test_dataset = loader.dataset(root_dir=dirname,mask=mask[0:1])
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=1, shuffle=False)

            for i, aa in enumerate(test_loader):                
                #print(i)
                batch1=aa['batch1']
                batch2=aa['batch2']
                label=aa['label']
                label.view(-1,1,1,1)
                batch1=batch1.cuda(0)
                batch2=batch2.cuda(0)
                loss= tracker.step(
                    batch1,batch2,labels=label, backward=False)
                loss=sigmoid(float(loss))
                responses.append(loss)         
                #print(responses) 
                
                if i%135==0:
                    cl.append(aa['cl'][0])
                    rf.append(aa['rf'][0])
            for j in range(len(rf)): 
                wbuffer=responses[j*135:135+j*135]                       
                wbuffer.insert(0,cl[j])
                wbuffer.insert(0,rf[j])  
                wr.writerow(wbuffer)
                wbuffer=[]                  
            f.close()
        mask = mask[1:] + mask[0:1]
