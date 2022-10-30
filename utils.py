#!/usr/bin/python
# Author: Siddhartha Gairola (t-sigai at microsoft dot com))
import numpy as np
import os
import io
import math
import random
import pandas as pd

import matplotlib.pyplot as plt
import librosa
import librosa.display
import cv2
import cmapy

import nlpaug
import nlpaug.augmenter.audio as naa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.signal import butter, lfilter





def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lwcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b,a, data)
    return y

def Extract_Annotation_Data(file_name, data_dir):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient Number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(data_dir, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return recording_info, recording_annotations

# get annotations data and filenames
def get_annotations(data_dir):
	filenames = [s.split('.')[0] for s in os.listdir(data_dir) if '.txt' in s]
	i_list = []
	rec_annotations_dict = {}
	for s in filenames:
		i,a = Extract_Annotation_Data(s, data_dir)
		i_list.append(i)
		rec_annotations_dict[s] = a

	recording_info = pd.concat(i_list, axis = 0)
	recording_info.head()

	return filenames, rec_annotations_dict

def slice_data(start, end, raw_data, sample_rate):
    max_ind = len(raw_data) 
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]

def get_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0:
        return 0
    elif crackle == 1 and wheeze == 0:
        return 1
    elif crackle == 0 and wheeze == 1:
        return 2
    else:
        return 3
#Used to split each individual sound file into separate sound clips containing one respiratory cycle each
#output: [filename, (sample_data:np.array, start:float, end:float, label:int (...) ]
#label: 0-normal, 1-crackle, 2-wheeze, 3-both
def get_sound_samples(recording_annotations, file_name, data_dir, sample_rate):
    sample_data = [file_name]
    # load file with specified sample rate (also converts to mono)
    data, rate = librosa.load(os.path.join(data_dir, file_name+'.wav'), sr=sample_rate)
    #print("Sample Rate", rate)
    
    for i in range(len(recording_annotations.index)):
        row = recording_annotations.loc[i]
        start = row['Start']
        end = row['End']
        crackles = row['Crackles']
        wheezes = row['Wheezes']
        audio_chunk = slice_data(start, end, data, rate)
        sample_data.append((audio_chunk, start,end, get_label(crackles, wheezes)))
    return sample_data


# split samples according to desired length
'''
types:
* 0: simply pad by zeros
* 1: pad with duplicate on both sides (half-n-half)
* 2: pad with augmented sample on both sides (half-n-half)	
'''
def split_and_pad(original, desiredLength, sample_rate, types=0):
	if types==0:
		return split_and_pad_old(original, desiredLength, sample_rate)

	output_buffer_length = int(desiredLength*sample_rate)
	soundclip = original[0].copy()
	n_samples = len(soundclip)

	output = []
	# if: the audio sample length > desiredLength, then split & pad
	# else: simply pad according to given type 1 or 2
	if n_samples > output_buffer_length:
		frames = librosa.util.frame(soundclip, frame_length=output_buffer_length, hop_length=output_buffer_length//2, axis=0)
		for i in range(frames.shape[0]):
			output.append((frames[i], original[1], original[2], original[3], original[4], i, 0))

		last_id = frames.shape[0]*(output_buffer_length//2)
		last_sample = soundclip[last_id:]; pad_times = (output_buffer_length-len(last_sample))/len(last_sample)
		padded = generate_padded_samples(soundclip, last_sample, output_buffer_length, sample_rate, types)
		output.append((padded, original[1], original[2], original[3], original[4], i+1, pad_times))

	else:
		padded = generate_padded_samples(soundclip, soundclip, output_buffer_length, sample_rate, types); pad_times = (output_buffer_length-len(soundclip))/len(soundclip)
		output.append((padded, original[1], original[2], original[3], original[4], 0, pad_times))

	return output

def split_and_pad_old(original, desiredLength, sample_rate):
    output_buffer_length = int(desiredLength * sample_rate)
    soundclip = original[0].copy()
    n_samples = len(soundclip)
    total_length = n_samples / sample_rate #length of cycle in seconds
    n_slices = int(math.ceil(total_length / desiredLength)) #get the minimum number of slices needed
    samples_per_slice = n_samples // n_slices
    src_start = 0 #Staring index of the samples to copy from the original buffer
    output = [] #Holds the resultant slices
    for i in range(n_slices):
        src_end = min(src_start + samples_per_slice, n_samples)
        length = src_end - src_start
        copy = generate_padded_samples_old(soundclip[src_start:src_end], output_buffer_length)
        output.append((copy, original[1], original[2], original[3], original[4], i))
        src_start += length
    return output

def generate_padded_samples_old(source, output_length):
    copy = np.zeros(output_length, dtype = np.float32)
    src_length = len(source)
    frac = src_length / output_length
    if(frac < 0.5):
        #tile forward sounds to fill empty space
        cursor = 0
        while(cursor + src_length) < output_length:
            copy[cursor:(cursor + src_length)] = source[:]
            cursor += src_length
    else:
        copy[:src_length] = source[:]
    return copy

def generate_padded_samples(original, source, output_length, sample_rate, types):
	copy = np.zeros(output_length, dtype=np.float32)
	src_length = len(source)
	left = output_length-src_length # amount to be padded
	# pad front or back
	prob = random.random()
	if types == 1:
		aug = original
	else:
		aug = gen_augmented(original, sample_rate)

	while len(aug) < left:
		aug = np.concatenate([aug, aug])

	if prob < 0.5:
		#pad back
		copy[left:] = source
		copy[:left] = aug[len(aug)-left:]
	else:
		#pad front
		copy[:src_length] = source[:]
		copy[src_length:] = aug[:left]

	return copy


#**********************DATA AUGMENTAION***************************
#Creates a copy of each time slice, but stretches or contracts it by a random amount
def gen_augmented(original, sample_rate):
	# list of augmentors available from the nlpaug library
	augment_list = [
	#naa.CropAug(sampling_rate=sample_rate)
	naa.NoiseAug(),
	naa.SpeedAug(),
	naa.LoudnessAug(factor=(0.5, 2)),
	naa.VtlpAug(sampling_rate=sample_rate, zone=(0.0, 1.0)),
	naa.PitchAug(sampling_rate=sample_rate, factor=(-1,3))
	]
	# sample augmentation randomly
	aug_idx = random.randint(0, len(augment_list)-1)
	augmented_data = augment_list[aug_idx].augment(original)
	return augmented_data

#Same as above, but applies it to a list of samples
def augment_list(audio_with_labels, sample_rate, n_repeats):
    augmented_samples = []
    for i in range(n_repeats):
        addition = [(gen_augmented(t[0], sample_rate), t[1], t[2], t[3], t[4]+i+1 ) for t in audio_with_labels]
        augmented_samples.extend(addition)
    return augmented_samples

def create_spectrograms(current_window, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512):
    fig = plt.figure(figsize=[1.0, 1.0])
    #fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    # There may be a better way to do the following, skipping it for now.
    buf = io.BytesIO()
    plt.savefig(buf, dpi=800, bbox_inches='tight',pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    plt.close('all')
    return img

def create_spectrograms_raw(current_window, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512):
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S-S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('viridis'))
    height, width, _ = img.shape
    img = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_LINEAR)
    return img

def create_mel_raw(current_window, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512, resz=1):
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S-S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('magma'))
    height, width, _ = img.shape
    if resz > 0:
        img = cv2.resize(img, (width*resz, height*resz), interpolation=cv2.INTER_LINEAR)
    img = cv2.flip(img, 0)
    return img

#Transpose and wrap each array along the time axis
def rollFFT(fft):
    n_row, n_col = fft.shape[:2]
    pivot = np.random.randint(n_col)
    return np.reshape(np.roll(fft, pivot, axis = 1), (n_row, n_col, 1))

def rollAudio(audio):
    # expect audio to be 1 dimensional
    pivot = np.random.randint(audio.shape[0])
    rolled_audio = np.roll(audio, pivot, axis=0)
    assert audio.shape[0] == rolled_audio.shape[0], "Roll audio shape mismatch"
    return rolled_audio


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

# save images from dataloader in dump_images/train or dump_images/test based on train_flag
# expect images to be in RGB format
# image is a tuple: (spectrogram, filename, label, cycle, split)
def save_images(image, train_flag):
    save_dir = 'dump_image'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    if train_flag:
        save_dir = os.path.join(save_dir, 'train')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, image[1]+'_'+str(image[2])+'_'+str(image[3])+'_'+str(image[4])+'.jpg'), cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR))
    else:
        save_dir = os.path.join(save_dir, 'test')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, image[1]+'_'+str(image[2])+'_'+str(image[3])+'_'+str(image[4])+'.jpg'), cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR))


#################
##for baselines##
#################

def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
       

    return sample_class_mean, precision
    
    
    
def get_Mahalanobis_score(model, test_loader, num_classes, outf, out_flag , sample_mean, precision, layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''

    model.eval()
    Mahalanobis = []
    
    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer_index))
        
    g = open(temp_file_name, 'w')
    
    for data, target in test_loader:
        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        #print(data)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
                
        gaussian_score, _ = torch.max(gaussian_score, dim=1)
        Mahalanobis.extend(gaussian_score.cpu().numpy())
        for i in range(data.size(0)):
            g.write("{}\n".format(gaussian_score[i]))
    g.close()
        
     

    return Mahalanobis
    
    
    
    

# others
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        #inputs = inputs[:,0,:,:,:]
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std    
    
class imbalance_data(Dataset):
    def __init__(self, imbal_class_prop, dataset):
            self.dataset = dataset
            self.imbal_class_prop = imbal_class_prop
            self.output_dim = len(imbal_class_prop)
            self.data = self.creat_imbalace()
            print(len(self.data))
            
            
    def creat_imbalace(self):
        # Get all targets and count the number of class instances
        targets = []
        datas = []
        for idex in range(len(self.dataset)):
            img, label = self.dataset[idex]
            targets.append(label)
            datas.append((img, label))
        
        targets = np.array(targets)
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        print(class_counts)

        # Create artificial imbalanced class counts
        imbal_class_counts = [int(num*4500) for num in self.imbal_class_prop]
        print(imbal_class_counts)

        # Get class indices
        class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
        #print(class_indices)

        # Get imbalanced number of instances
        #print('---------------')
        imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
        #print(imbal_class_indices)
        imbal_class_indices = np.hstack(imbal_class_indices)
        imbal_class_indices= imbal_class_indices.tolist()

        # Set target and data to dataset
        return [datas[i] for i in imbal_class_indices]

    def __getitem__(self, index):
        img, target = self.data[index]
        return img, target
        
    def __len__(self):
        return len(self.data)
        
        
class re_balance_data(Dataset):
    def __init__(self, imbal_class_prop, dataset):
            self.dataset = dataset
            self.imbal_class_prop = imbal_class_prop
            self.output_dim = len(imbal_class_prop)
            self.data = self.creat_balace()
            print(len(self.data))
            
            
    def creat_balace(self):
        # Get all targets and count the number of class instances
        targets = []
        datas = []
        for idex in range(len(self.dataset)):
            img, label = self.dataset[idex]
            targets.append(label)
            datas.append((img, label))
        
        targets = np.array(targets)
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        print(class_counts)
        class_count = max(class_counts)

        # Get class indices
        class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
        #print(class_indices)

        # Get imbalanced number of instances
        #print('---------------')
        imbal_class_indices = []
        for class_idx in class_indices:
            np.random.seed(0)
            imbal_class_indices.append(np.random.choice(class_idx,class_count))

        imbal_class_indices = np.hstack(imbal_class_indices)
        imbal_class_indices= imbal_class_indices.tolist()

        # Set target and data to dataset
        return [datas[i] for i in imbal_class_indices]

    def __getitem__(self, index):
        img, target = self.data[index]
        return img, target
        
    def __len__(self):
        return len(self.data)