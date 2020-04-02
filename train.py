from model import DeepDual,DeepDualNet
from torch.utils.data import DataLoader
import loader as loader
import os
import sys
import csv
import time
import numpy
import torch
import torch.nn.functional as F

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+numpy.exp(-x))

csv_path='sim'
if not os.path.exists(csv_path):
    os.makedirs(csv_path)
loss_path='./loss/loss.csv'
if not os.path.exists('./loss/'):
    os.makedirs('./loss/')
    f = open(loss_path, 'w', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(["Epoch","NetLoss"])
    f.close()



if __name__ == '__main__':
    # setup dataset
    dirname='../../2_1.DCVL_Face_DB/'           # dir of dataset
    cuda = torch.cuda.is_available()
    net_dir = 'model'                       # dir where model will be saved
    DeepDual = DeepDual()
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)
    epoch_num = 100
    mask=list(range(1,51))
    for trial in range(50):
        if mask[0]>0:
            net_path=os.path.join(net_dir,'model_%02d.pth'%(trial))
            train_dataset = loader.dataset(root_dir=dirname,mask=mask[1:])
            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1000,
                            shuffle=True,pin_memory=cuda, drop_last=True, num_workers=4)
            t=time.perf_counter()
            
            for epoch in range(epoch_num):
                mean_loss=0
                cnt=0
                for step, sample in enumerate(train_loader):
                    # bring the label, left, and right images from the loader
                    label=sample['label']
                    label.view(-1,1,1,1)
                    left_img=sample['left_img']
                    right_img=sample['right_img']
                    left_img=left_img.cuda(0)
                    right_img=right_img.cuda(0)
                    label=label.cuda(0)
                    # put the input to the model and get the loss
                    loss= DeepDual.step(
                        left_img,right_img,labels=label, backward=True, update_lr=(step == 0))
                    cnt+=1
                    mean_loss +=loss
                    sys.stdout.flush()

                    if step%1 == 0:
                        old_t=t
                        t=time.perf_counter()
                        print('Trial:{} Epoch [{}][{}/{}]: Loss: {:.12f}, time: {:.2f}'.format(trial+1, epoch + 1, step + 1, len(train_loader), loss,(t-old_t)))
                # write the loss in a csv file
                f = open(loss_path, 'a', encoding='utf-8')
                wr = csv.writer(f)
                wr.writerow([epoch,(mean_loss/cnt)])
                f.close()
            # save the model after finishing training
            torch.save(DeepDual.net.state_dict(), net_path)
            
            # After finishing the training, we test trained model and calculate the EER
            responses=[]
            cl=[]
            rf=[]
            wbuffer=[]
            f = open('./sim/similarity_%02d.csv'%(trial), 'a', encoding='utf-8')
            wr = csv.writer(f,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            test_dataset = loader.dataset(root_dir=dirname,mask=mask[0:1])
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=1, shuffle=False)
            for i, sample in enumerate(test_loader):
                print(i)
                left_img=sample['left_img']
                right_img=sample['right_img']
                label=sample['label']
                label.view(-1,1,1,1)
                left_img=left_img.cuda(0)
                right_img=right_img.cuda(0)
                loss= DeepDual.step(
                    left_img,right_img,labels=label, backward=False)
                loss=sigmoid(float(loss))
                responses.append(loss)         
                if i%135==0:
                    cl.append(sample['cl'][0])
                    rf.append(sample['rf'][0])
            for j in range(len(rf)): 
                wbuffer=responses[j*135:135+j*135]                       
                wbuffer.insert(0,cl[j])
                wbuffer.insert(0,rf[j])  
                wr.writerow(wbuffer)
                wbuffer=[]                  
            f.close()
            
        mask = mask[1:] + mask[0:1]
