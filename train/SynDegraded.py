# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:54:23 2025

@author: Jeremy
"""


import cv2
import numpy as np
import os

def create_glow(img,intmax):

    w,h,c = img.shape
    whmin = 512#np.min(w,h)
	
    glow_center = (np.random.randint(100,400),np.random.randint(100,400))
    glow_radius = np.random.randint(50,300)#np.random.randint(whmin//10,whmin//3)
    intensity = np.random.uniform(0.05,intmax)
    glow_color=(255, 255, 255)
	
    glow_layer = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(glow_layer, glow_center, glow_radius, glow_color, -1)
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), glow_radius / 3)
    
    result = cv2.addWeighted(img.astype(np.uint8), 1.0, glow_layer.astype(np.uint8), intensity, 0)
    
    return result

def SyntheticDeg(img,sor):

	tt = np.random.uniform(0.50, 0.85)    
	L = np.random.uniform(0.50, 0.90)
	G = np.random.uniform(1.00, 2.0)
	A = np.random.uniform(0.85, 0.95)
	degradedimg = np.zeros((img.shape))
	
	if sor == 0:    #Haze	
		degradedimg = img * tt + A * (1 - tt)
	elif sor == 1:    #Low	
		sors = np.random.randint(0,2)	
		if sors == 0:
			degradedimg = img * L
		else:
			degradedimg = np.power(img,G)				
	elif sor == 2:    #Glow		
		for i in range(3):
			degradedimg = create_glow(img*255,intmax=0.50)/255
	elif sor == 3:    #Haze	and Low	
		sors = np.random.randint(0,2)	
		if sors == 0:
			degradedimg = img * L * tt + A * (1 - tt)
		else:
			degradedimg = np.power(img,G) * tt + A * (1 - tt)	
	elif sor == 4:    #Glow		
		for i in range(3):
			degradedimg = create_glow((img * tt + A * (1 - tt))*255,intmax=0.25)/255
	elif sor == 5:    #Low and Glow
		sors = np.random.randint(0,2)	
		if sors == 0:
			degradedimg = create_glow(img * L*255,intmax=0.75)/255
		else:
			degradedimg = create_glow(np.power(img,G)*255,intmax=0.75)/255
	else:    #Haze, Low, and Glow			
		sors = np.random.randint(0,2)	
		if sors == 0:
			degradedimg = create_glow((img * L * tt + A * (1 - tt))*255,intmax=0.25)/255
		else:
			degradedimg = create_glow((np.power(img,G) * tt + A * (1 - tt))*255,intmax=0.25)/255
					    
	return degradedimg

files = os.listdir('./T2O_Clear')
totalfiles = len(files)

for i in range(totalfiles):
	print("%d/%d"%(i,totalfiles))
	img = cv2.resize(cv2.imread('./T2O_Clear/'+files[i]),(512,512))/255
	degraded_img = SyntheticDeg(img,sor=0)
			
	cv2.imwrite('./T2O_Haze/' + files[i],degraded_img*255)

for i in range(totalfiles):
	print("%d/%d"%(i,totalfiles))
	img = cv2.resize(cv2.imread('./T2O_Clear/'+files[i]),(512,512))/255
	degraded_img = SyntheticDeg(img,sor=1)
			
	cv2.imwrite('./T2O_Low/' + files[i],degraded_img*255)

for i in range(totalfiles):
	print("%d/%d"%(i,totalfiles))
	img = cv2.resize(cv2.imread('./T2O_Clear/'+files[i]),(512,512))/255
	degraded_img = SyntheticDeg(img,sor=2)
			
	cv2.imwrite('./T2O_Glow/' + files[i],degraded_img*255)

for i in range(totalfiles):
	print("%d/%d"%(i,totalfiles))
	img = cv2.resize(cv2.imread('./T2O_Clear/'+files[i]),(512,512))/255
	degraded_img = SyntheticDeg(img,sor=3)
			
	cv2.imwrite('./T2O_HazeLow/' + files[i],degraded_img*255)
	
for i in range(totalfiles):
	print("%d/%d"%(i,totalfiles))
	img = cv2.resize(cv2.imread('./T2O_Clear/'+files[i]),(512,512))/255
	degraded_img = SyntheticDeg(img,sor=4)
			
	cv2.imwrite('./T2O_HazeGlow/' + files[i],degraded_img*255)
	
for i in range(totalfiles):
	print("%d/%d"%(i,totalfiles))
	img = cv2.resize(cv2.imread('./T2O_Clear/'+files[i]),(512,512))/255
	degraded_img = SyntheticDeg(img,sor=5)
			
	cv2.imwrite('./T2O_LowGlow/' + files[i],degraded_img*255)
	
for i in range(totalfiles):
	print("%d/%d"%(i,totalfiles))
	img = cv2.resize(cv2.imread('./T2O_Clear/'+files[i]),(512,512))/255
	degraded_img = SyntheticDeg(img,sor=6)
			
	cv2.imwrite('./T2O_HazeLowGlow/' + files[i],degraded_img*255)
	
		

