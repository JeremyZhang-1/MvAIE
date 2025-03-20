# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:54:23 2025

@author: Jeremy
"""

from makedataset import *
import argparse

if __name__ == "__main__":
		
	parser = argparse.ArgumentParser(description="Building the training patch database")
	
	parser.add_argument("--rgb", action='store_true',default = True, help='prepare RGB database instead of grayscale')
	parser.add_argument("--patch_size", "--p", type=int, default=256, help="Patch size")
	parser.add_argument("--stride", "--s", type=int, default=240, help="Size of stride")
	args = parser.parse_args()

	TrainSynImg('./dataset/Train/T2O_Clear','./dataset/Train/T2O_Haze','./dataset/Train/T2O_Low','./dataset/Train/T2O_Glow','./dataset/Train/T2O_HazeLow','./dataset/Train/T2O_HazeGlow','./dataset/Train/T2O_LowGlow','./dataset/Train/T2O_HazeLowGlow',args.patch_size,args.stride)
