# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:25:06 2022

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import torchvision.transforms as transformtransforms
from tqdm import tqdm
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader

EPOCH = 20
BATCH = 20
NUM_WORKERS = 0

SEED = 2099

image_size = 64
LR = 0.0002
NOISE = 100


time_start_all = time.time()
# #set manual seed to a constant get a consistent output
# random.seed(SEED)
# torch.manual_seed(SEED)
# print("Random Seed: ", SEED)



print('torch.version: ',torch. __version__)
print('torch.version.cuda: ',torch.version.cuda)
print('torch.cuda.is_available: ',torch.cuda.is_available())
print('torch.cuda.device_count: ',torch.cuda.device_count())
print('torch.cuda.current_device: ',torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))
device = torch.device("cuda")
# print(torch.cuda.get_arch_list())





#loading the dataset
transform=transforms.Compose([transforms.Resize(image_size),
                              transforms.CenterCrop(image_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5)),
                              ])


dataset = torchvision.datasets.CIFAR10(root="./data/", 
                                       download=True,
                                       transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=BATCH,
                                         shuffle=True, 
                                         num_workers=NUM_WORKERS)


print(dataset.classes)
print(dataset.data.shape)



class Generator(nn.Module):
    def __init__(self, noise = NOISE):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

    def initialize_weights(self, w_mean=0, w_std=0.02, b_mean=1, b_std=0.02):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, w_mean, w_std)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, b_mean, b_std)
                nn.init.constant_(m.bias.data, 0)

device = torch.device("cuda")
G = Generator().to(device)
summary(G, input_size=(100,1,1))




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def initialize_weights(self, w_mean=0, w_std=0.02, b_mean=1, b_std=0.02):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, w_mean, w_std)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, b_mean, b_std)
                nn.init.constant_(m.bias.data, 0)

device = torch.device("cuda")
D = Discriminator().to(device)
summary(D, input_size=(3,64,64))



def norm(data):
    data = ((data-np.min(data))/(np.max(data)-np.min(data))*255).astype(np.uint8)
    return data

def Image_wall(images):
    image = images[random.randint(0, 9)].detach().permute(1,2,0).cpu().numpy()
    frames_1 = np.array([])
    for i in range(0,5):
        frame_i = images[i].detach().permute(1,2,0).cpu().numpy()
        if i == 0: frames_1 = frame_i
        else: frames_1 = np.hstack((frames_1,frame_i))
        
    frames_2 = np.array([])            
    for i in range(5,10):
        frame_i = images[i].detach().permute(1,2,0).cpu().numpy()
        if i == 5: frames_2 = frame_i
        else: frames_2 = np.hstack((frames_2,frame_i))
        
    frames = np.vstack((frames_1,frames_2))
    # frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    frames = norm(frames)
    image = norm(image)
    return frames, image


'''
loss & optimizer
'''
torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Using RMSprop

# optimizerG = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
# optimizerD = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_G = optim.RMSprop(G.parameters(), lr=LR)
optimizer_D = optim.RMSprop(D.parameters(), lr=LR)


# criterion = nn.BCELoss()

input_noise = torch.randn(BATCH, NOISE, 1, 1)
input_noise = input_noise.to(device)

real_label = 1.0
fake_label = 0.0



'''
Training
'''
D_period = 1
G_period = 1
G_loss_list = []
D_loss_list  = []
image_list = []
frame_list = []

print('start training...')
print('Epochs: %d | Batch Size: %d | Learning Rate: %.4f' % (EPOCH,BATCH,LR))
G.train(True)
D.train(True)
time_start = time.time()

for epoch in range(EPOCH):
    for i, data in enumerate(dataloader):
        
        '''
        Train D
        '''
        for k in range(D_period):
            optimizer_D.zero_grad()
            noise = torch.randn(BATCH, NOISE, 1, 1, device=device)
    
            real_data = data[0].to(device)
            fake_data = G(noise)
            
            loss_real = D(real_data)
            loss_fake = D(fake_data)
            # loss_real = criterion(output, label)
            
            #Wasserstein
            # G_loss = -torch.mean(loss_fake)
            D_loss = -torch.mean(loss_real-loss_fake)
            
            D_loss.backward()
            optimizer_D.step()
            # D_x = output.mean().item()
            
            for p in D.parameters():
                p.data.clamp_(-0.01,0.01)
    
    
        '''
        Train G
        '''
        for m in range(G_period):
            optimizer_G.zero_grad()
            
            noise = torch.randn(BATCH, NOISE, 1, 1, device=device)
            fake_data = G(noise)
            
            loss_fake = D(fake_data)
            #Wasserstein
            G_loss = -torch.mean(loss_fake)
            
            G_loss.backward()
            optimizer_G.step()
            
            
            
        G_loss_list.append(G_loss.item())
        D_loss_list.append(D_loss.item())

            
        if i % (len(dataloader)//25) == 0:
            D.eval()
            G.eval()
            
            noise = torch.randn(10, NOISE, 1, 1, device=device)
            fake_data = G(noise)
            
            fake_images, G_image = Image_wall(fake_data)
            real_images,_ = Image_wall(real_data)
            frames = np.vstack((real_images,fake_images))
            
            plt.figure(figsize=(10,10))
            plt.imshow(frames)
            plt.axis('off')
            plt.title('Epoch: %d niter: %d'%(epoch,i),fontsize=16)
            plt.show()
            
            # frame_list.append(G_image)
            # image_list.append(fake_images)
            
            time_end = time.time()
            time_cost = time_end - time_start
            time_start = time.time()
            
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
                epoch+1, EPOCH, i, len(dataloader), D_loss.item(), G_loss.item()))

            D.train()
            G.train()


    #save for every epoch
    plt.figure(figsize=(20,10))
    plt.plot(G_loss_list, label='G')
    plt.plot(D_loss_list, label='D')
    plt.xlabel('iter',fontsize=20)
    plt.ylabel('loss',fontsize=20)
    plt.title('Training',fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('WGAN_loss.png')
    plt.show()
    torch.save(G.state_dict(), './weights/WGAN_G_test.pth')
    torch.save(D.state_dict(), './weights/WGAN_D_test.pth')




#save model
time_end_all = time.time()
print('Done! Totoal time cost: ',time_end_all-time_start_all)
t = time.strftime("%Y%m%d_%H%M", time.localtime()) 
torch.save(G.state_dict(), './weights/WGAN_G_'+t+'.pth')
torch.save(D.state_dict(), './weights/WGAN_D_'+t+'.pth')


def show_images(frame_list):
    n = len(frame_list)

    new_frame_list = []
    interval = n//EPOCH
    for i in range(0,n,interval): #print(i)
        index = random.randint(i, i+interval-1)
        new_frame_list.append(frame_list[index])
        
    n = len(new_frame_list) 
    col = 10
    row = n//col
    
    frames = np.array([])
    for i in range(row):
        frame_row = np.array([])
        for j in range(col):
            frame_i = new_frame_list[i*col + j]
            if j==0: frame_row = frame_i
            else: frame_row = np.hstack((frame_row,frame_i))
        
        if i==0: frames = frame_row
        else: frames = np.vstack((frames,frame_row))
    
    return frames
   
frames = show_images(frame_list)     

plt.figure(figsize=(10,10))
plt.imshow(frames)
plt.axis('off')
plt.title('WGAN Epochs:%d Batch:%d D/G: %d/%d' % (
            epoch,BATCH,D_period,G_period),fontsize=20)
plt.legend(fontsize=20)
plt.savefig('WGAN_image_wall_plot.png')
plt.show() 

cv2.imwrite('WGAN_image_wall.png', cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))





'''
test model
get 10 best images
'''
# D.load_state_dict(torch.load('./weights/WGAN_D_test.pth'))
# G.load_state_dict(torch.load('./weights/WGAN_G_test.pth'))

def test_model(num_image):
    D.eval()
    G.eval()
    
    best_image = []
    score_list = []
    index = 0
    
    # for i in range(100):
    while index < num_image:
        noise = torch.randn(1, NOISE, 1, 1, device=device)
        test_image = G(noise)
        score = D(test_image).detach().cpu().numpy()[0][0][0][0]
        # image = test_image.detach().permute(0,2,3,1).cpu().numpy()[0]
        # best_image.append(norm(image))
        score_list.append(score)
        if score <0.2:
            print(score)
            image = test_image.detach().permute(0,2,3,1).cpu().numpy()[0]
            best_image.append(norm(image))
            index = index +1

    return best_image,score_list

best_image,score_list = test_model(10)
plt.plot(score_list)


def show_images(image_list):
    n = len(image_list)

    col = 5
    row = n//col
    
    frames = np.array([])
    for i in range(row):
        frame_row = np.array([])
        for j in range(col):
            frame_i = image_list[i*col + j]
            if j==0: frame_row = frame_i
            else: frame_row = np.hstack((frame_row,frame_i))
        
        if i==0: frames = frame_row
        else: frames = np.vstack((frames,frame_row))
    
    return frames
   
frames = show_images(best_image)      

plt.figure(figsize=(10,10))
plt.imshow(frames)
plt.axis('off')
plt.title('WGAN Best Epochs:%d Batch:%d D/G: %d/%d' % (
            epoch+1,BATCH,1,1),fontsize=20)
plt.legend(fontsize=20)
plt.savefig('WGAN_image_wall_plot.png')
plt.show() 

cv2.imwrite('WGAN_image_wall.png', cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))























