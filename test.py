# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:54:23 2025

@author: Jeremy
"""


import torch
import torch.nn as nn

#from tensorboardX import SummaryWriter
import numpy as np
import cv2
import time
import os
from MvAIENet import *
import utils_train


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(checkpoint_dir,IsGPU):
    
	if IsGPU == 1:
		model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
		net = MvAIENet()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']


	return model, optimizer,cur_epoch

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			print( param_group['lr'])
	return optimizer

def train_psnr(train_in,train_out):
	
	psnr = utils_train.batch_psnr(train_in,train_out,1.)
	return psnr


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])
	

def GammaCorrection(img,gamma1,gamma2,istrain=True):
    if istrain is True:
        img_g1 = np.power(img,gamma1)
        img_g2 = np.power(img,gamma2)
        
        return torch.cat((img_g1, img_g2),0)
    
    else:
        img = img / 255.0
        img_g1 = np.power(img,gamma1) * 255
        img_g2 = np.power(img,gamma2) * 255
        
        return np.concatenate((img_g1, img_g2), axis=-1)
        

def HSVCorrection(img,value1,value2,istrain=True):
    if istrain is True:
        img = chw_to_hwc(img.numpy())*255

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        h, s, v = cv2.split(img)
        
        v1 = np.clip(v * value1, 0, 255)#.astype(np.uint8)
        v2 = np.clip(v * value2, 0, 255)#.astype(np.uint8)
        
        hsv1 = cv2.merge([h, s, v1])
        hsv2 = cv2.merge([h, s, v2])    
    
        img_hsv1 = hwc_to_chw(cv2.cvtColor(hsv1, cv2.COLOR_HSV2BGR)/255.0)
        img_hsv2 = hwc_to_chw(cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)/255.0)
        
        return torch.cat((torch.from_numpy(img_hsv1), torch.from_numpy(img_hsv2)),0) 
        
    else:
        img = cv2.cvtColor(imgrgb, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        
        v1 = np.clip(v * value1, 0, 255).astype(np.uint8)
        v2 = np.clip(v * value2, 0, 255).astype(np.uint8)
        
        hsv1 = cv2.merge([h, s, v1])
        hsv2 = cv2.merge([h, s, v2])    
    
        img_hsv1 = cv2.cvtColor(hsv1, cv2.COLOR_HSV2BGR)
        img_hsv2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)  
        
        return np.concatenate((img_hsv1, img_hsv2), axis=-1)
    
	
if __name__ == '__main__': 	
	checkpoint_dir = './checkpoint/'
	test_dir = './input'
	result_dir = './output'    
	testfiles = os.listdir(test_dir)
    
	IsGPU = 1    #GPU is 1, CPU is 0

	print('> Loading dataset ...')

	lr_update_freq = 30
	model,optimizer,cur_epoch = load_checkpoint(checkpoint_dir,IsGPU)

	if IsGPU == 1:
		for f in range(len(testfiles)):
			model.eval()
			with torch.no_grad():
				imgrgb = cv2.imread(test_dir + '/' + testfiles[f])
				imggamma = GammaCorrection(imgrgb,gamma1=2/3,gamma2 =3/2,istrain=False)
				imghsv = HSVCorrection(imgrgb,value1=2/3,value2 =3/2,istrain=False)
                       
                
				imgrgb   = hwc_to_chw(imgrgb / 255.0)
				imggamma = hwc_to_chw(imggamma / 255.0) 
				imghsv   = hwc_to_chw(imghsv / 255.0)                
                
				input_l = torch.from_numpy(imgrgb.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
				input_g = torch.from_numpy(imggamma.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
				input_h = torch.from_numpy(imghsv.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
                
				s = time.time()
				_,_,_,e_out = model(input_l,input_g,input_h,True,sor=7)
				e = time.time()   
	             		
				e_out = chw_to_hwc(e_out.squeeze().cpu().detach().numpy())	
                
				cv2.imwrite(result_dir + '/' + testfiles[f][:-4] +'_A'+'.png',np.clip(e_out*255,0.0,255.0))


                
	  
				
			
			

