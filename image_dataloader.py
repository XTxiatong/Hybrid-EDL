#!/usr/bin/python                                                       
# Author: Siddhartha Gairola (t-sigai at microsoft dot com)

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
 

import librosa
#from tqdm import tqdm

from utils import *
from fastai.vision import cutout

def spec_augment(image, num_mask=1, freq_masking_max_percentage=0.01, time_masking_max_percentage=0.01):
    spec = torch.empty_like(image).copy_(image)
    for i in range(num_mask):
        _, all_frames_num, all_freqs_num = spec.shape
        freq_percentage = freq_masking_max_percentage

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:,:, f0 : f0 + num_freqs_to_mask] = 0

        time_percentage = time_masking_max_percentage

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:,t0 : t0 + num_frames_to_mask, :] = 0
    return spec
    
    
def spec_noise(image, std=0.05):
    spec = torch.empty_like(image).copy_(image)
    #print(spec)
    noise = torch.randn(spec.shape)
    spec =  spec + noise*0.1
    #print(spec)
    return spec   


class image_loader(Dataset):
    def __init__(self, class_number, data_dir, folds_file, fold_list, train_flag, params_json, 
        input_transform=None, stetho_id=-1, aug_scale=None, is_OE=False, aug_type=None, aug_seed=None):
        # getting device-wise information
        self.class_number = class_number
        self.test_aug = aug_type
        self.aug_seed = aug_seed
        self.output_dim = class_number
        self.train_flag = train_flag
        self.data_dir = data_dir
        self.input_transform = input_transform
        
        # get patients dict in current fold based on train flag
        all_patients = open(folds_file)
        patient_dict = {}
        patient_list = []
        for line in all_patients:
            idx, disease, fold = line.strip().split(',')
            patient_dict[idx] = int(disease)
            if int(fold) in fold_list: #updated here
                patient_list.append(idx)       

        self.image_data = [] # each sample is a tuple with id_0: image_data, id_1: label
        self.labels = []
        files = os.listdir(data_dir)
        #for file in tqdm(files):
        for file in files:
           if file.split('.')[0] in patient_list:
               image = cv2.imread(os.path.join(data_dir,file))
               if '2017' in data_dir:
                    #print(image.shape)
                    image = cv2.resize(image, (767,1022))
               label = patient_dict[file.split('.')[0]]
               self.image_data.append([image,label])
               self.labels.append(label)
   
        # concatenation based augmentation scheme
        if train_flag and aug_scale:
            print("LEN AUDIO DATA", len(self.image_data))
            print('start augmentation')
            self.new_augment(scale=float(aug_scale))

        self.class_probs = np.zeros(class_number)
        for idx, sample in enumerate(self.image_data):
            self.class_probs[sample[1]] += 1.0
        if self.train_flag:
            print("TRAIN DETAILS")
        else:
            print("TEST DETAILS")
         
        print("CLASSWISE SAMPLE COUNTS:", self.class_probs)
        self.class_probs = self.class_probs / sum(self.class_probs)
        print("CLASSWISE PROBS", self.class_probs)
        print("LEN AUDIO DATA", len(self.image_data))

    def new_augment(self, scale=1):
        normal_n = [i for i in range(len(self.labels)) if self.labels[i]==0]   #number of the classs 0
        for c in range(1,self.class_number):
            index = [i for i in range(len(self.labels)) if self.labels[i]==c]
            aug_n = int(scale*(len(normal_n)-len(index)))
            index_aug = np.random.choice(index,aug_n)
            for idx in index_aug:
                image,label = self.image_data[idx][0],self.image_data[idx][1]
                image_aug = cv2.convertScaleAbs(image, alpha=0.92, beta=0.07)
                self.image_data.append([image_aug,label])
                self.labels.append(label)
                

    def __getitem__(self, index):

        image,label = self.image_data[index][0],self.image_data[index][1]
        if self.test_aug == 'Color':
            image = cv2.convertScaleAbs(image, alpha=0.95, beta=0.05)
        # apply image transform 
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.test_aug == 'Mask':
           image = spec_augment(image)
        if self.test_aug == 'Gaussian':
           image = spec_noise(image)
  
        
        return image, label

    def __len__(self):
        return len(self.image_data)
