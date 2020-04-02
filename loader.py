import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os, os.path
import glob
from PIL import Image
class dataset(Dataset):
    def __init__(self, root_dir,mask):
        self.root_dir = root_dir
        self.mask=mask
        self.transform= transforms.Compose([
            transforms.Resize((128,128), interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.NOF = 0
        self.list_left=[]
        self.list_right=[]
        self.list_disparity=[]
        self.image_processing()

    def image_processing(self):
        print(self.mask)
        for m in self.mask:
            list_left = sorted(glob.glob(os.path.join(self.root_dir,'*',str(m)+'_cropped','left*')))
            list_right = sorted(glob.glob(os.path.join(self.root_dir,'*',str(m)+'_cropped','right*')))
            self.list_left=self.list_left+list_left
            self.list_right=self.list_right+list_right
    def __len__(self):
        return len(self.list_left)

    def __getitem__(self, idx):
        left_img = Image.open(self.list_left[idx])
        right_img = Image.open(self.list_right[idx])
        rf=self.list_left[idx].split(os.sep)[3] # real or fake
        cl=self.list_left[idx].split(os.sep)[4][:-8] # mask (1~22)
        if rf=='real':
            label=torch.ones(1)
        else:
            label=torch.zeros(1)
        left_img=self.transform(left_img)
        right_img=self.transform(right_img)
        aa={'left_img':left_img,'right_img':right_img,'label':label,'rf':rf,'cl':cl}
        return aa

