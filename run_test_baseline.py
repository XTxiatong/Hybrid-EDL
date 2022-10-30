import logging
import os
import itertools
import argparse
import random
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10

from utils import *
from image_dataloader import *
from PosteriorNetwork import PosteriorNetwork
from train import train, train_sequential
from test import test, test_baseline

# input argmuments
parser = argparse.ArgumentParser(description='RespireNet: Lung Sound Classification')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.001,help='weight decay value')
parser.add_argument('--gpu_ids', default=[0], help='a list of gpus')
parser.add_argument('--num_worker', default=4, type=int, help='numbers of worker')
parser.add_argument('--batch_size', default=64, type=int, help='bacth size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--start_epochs', default=0, type=int, help='start epochs')
parser.add_argument('--class_number', default=7, type=int, help='number of classes')

parser.add_argument('--data_dir', default='../data/HAM10000_images', type=str, help='data directory')
parser.add_argument('--ood_data_dir', default='../data/ISIC-2017_Training_Data', type=str, help='data directory')
parser.add_argument('--ood_data_dir2', default='../data/', type=str, help='data directory')
parser.add_argument('--folds_file', default='../data/HAM10000_split.txt', type=str, help='folds text file')
parser.add_argument('--ood_folds_file', default='../data/ISIC-2017_split.txt', type=str, help='folds text file')
parser.add_argument('--train_fold', default=[0], type=int, help='Test Fold ID')
parser.add_argument('--vad_fold', default=[2], type=int, help='Test Fold ID')
parser.add_argument('--test_fold', default=[5,6], type=int, help='Test Fold ID')
parser.add_argument('--stetho_id', default=-1, type=int, help='Stethoscope device id')
parser.add_argument('--aug_scale', default=None, type=float, help='Augmentation multiplier')
parser.add_argument('--model_path',type=str, help='model saving directory')
parser.add_argument('--checkpoint', default=None, type=str, help='load checkpoint')
parser.add_argument('--test_only', default=False, type=bool, help='load checkpoint for testing')
parser.add_argument('--loss', default='UCE', type=str, help='loss function')
parser.add_argument('--latent_dim', default=6, type=int, help='latend dim')
parser.add_argument('--density_dim', default=6, type=int, help='density deep')
parser.add_argument('--no_density', default=False, type=bool, help='no flow')
parser.add_argument('--name', default='Flow', type=str, help='model save name')
args = parser.parse_args()

# Dataset parameters
seed_dataset=1
directory_dataset=args.data_dir
dataset_name=args.name #not used indeed
ood_dataset_names=['OOD_test']

# Architecture parameters
seed_model=123
directory_model='./saved_models'
architecture='densenet' #update
input_dims=[1,1,1] #update 
output_dim=args.class_number #update
hidden_dims=[64,64,64] #update
kernel_dim=None
latent_dim=args.latent_dim #need to specify
no_density=args.no_density 
density_type='radial_flow'
n_density=args.density_dim
k_lipschitz=None
budget_function='id'

# Training parameters
directory_results='./saved_results'
max_epochs=200
patience=30
frequency=2
batch_size=64
lr=args.lr
loss=args.loss
training_mode='joint'
regr=1e-5

##not used 
unscaled_ood=False
split=[0.75] # updated 
transform_min=args.aug_scale
transform_max=args.aug_scale


def run(
        # Dataset parameters
        seed_dataset,  # Seed to shuffle dataset. int
        directory_dataset,  # Path to dataset. string
        dataset_name,  # Dataset name. string
        ood_dataset_names,  # OOD dataset names.  list of strings
        unscaled_ood,  # If true consider also unscaled versions of ood datasets. boolean
        split,  # Split for train/val/test sets. list of floats
        transform_min,  # Minimum value for rescaling input data. float
        transform_max,  # Maximum value for rescaling input data. float

        # Architecture parameters
        seed_model,  # Seed to init model. int
        directory_model,  # Path to save model. string
        architecture,  # Encoder architecture name. string
        input_dims,  # Input dimension. List of ints
        output_dim,  # Output dimension. int
        hidden_dims,  # Hidden dimensions. list of ints
        kernel_dim,  # Input dimension. int
        latent_dim,  # Latent dimension. int
        no_density,  # Use density estimation or not. boolean
        density_type,  # Density type. string
        n_density,  # Number of density components. int
        k_lipschitz,  # Lipschitz constant. float or None (if no lipschitz)
        budget_function,  # Budget function name applied on class count. name

        # Training parameters
        directory_results,  # Path to save resutls. string
        max_epochs,  # Maximum number of epochs for training
        patience,  # Patience for early stopping. int
        frequency,  # Frequency for early stopping test. int
        batch_size,  # Batch size. int
        lr,  # Learning rate. float
        loss,  # Loss name. string
        training_mode,  # 'joint' or 'sequential' training. string
        regr):  # Regularization factor in Bayesian loss. float

    logging.info('Received the following configuration:')
    logging.info(f'DATASET | '
                 f'seed_dataset {seed_dataset} - '
                 f'dataset_name {dataset_name} - '
                 f'ood_dataset_names {ood_dataset_names} - '
                 f'split {split} - '
                 f'transform_min {transform_min} - '
                 f'transform_max {transform_max}')
    logging.info(f'ARCHITECTURE | '
                 f' seed_model {seed_model} - '
                 f' architecture {architecture} - '
                 f' input_dims {input_dims} - '
                 f' output_dim {output_dim} - '
                 f' hidden_dims {hidden_dims} - '
                 f' kernel_dim {kernel_dim} - '
                 f' latent_dim {latent_dim} - '
                 f' no_density {no_density} - '
                 f' density_type {density_type} - '
                 f' n_density {n_density} - '
                 f' k_lipschitz {k_lipschitz} - '
                 f' budget_function {budget_function}')
    logging.info(f'TRAINING | '
                 f' max_epochs {max_epochs} - '
                 f' patience {patience} - '
                 f' frequency {frequency} - '
                 f' batch_size {batch_size} - '
                 f' lr {lr} - '
                 f' loss {loss} - '
                 f' training_mode {training_mode} - '
                 f' regr {regr}')

    ##################
    ## Load dataset ##
    ##################
 
    mean, std = [0.5706, 0.5464, 0.7636], [0.1329, 0.1182, 0.0895]
    input_transform = Compose([ToTensor(), Normalize(mean, std)])
    train_dataset = image_loader(args.class_number,args.data_dir, args.folds_file, args.train_fold, 
                True, "params_json", input_transform, stetho_id=args.stetho_id, aug_scale=args.aug_scale)
    N = [train_dataset.class_probs[i]*len(train_dataset.image_data) for i in range(output_dim)]
    N = torch.tensor(N)  
    vad_dataset = image_loader(args.class_number,args.data_dir, args.folds_file, args.vad_fold, 
                False, "params_json", input_transform, stetho_id=args.stetho_id, aug_scale=args.aug_scale)        
    test_dataset = image_loader(args.class_number,args.data_dir, args.folds_file, args.test_fold, 
                False, "params_json", input_transform, stetho_id=args.stetho_id)
    
    ood_dataset = image_loader(args.class_number,args.ood_data_dir, args.ood_folds_file, [0], 
            False, "params_json", input_transform, stetho_id=args.stetho_id)
                    
    ood_dataset2 = CIFAR10(root='../data', train=False,download=True, transform=input_transform)
    
    # weighted sampler
    reciprocal_weights = []
    for idx in range(len(train_dataset)):
        reciprocal_weights.append(train_dataset.class_probs[train_dataset.labels[idx]])
    weights = (1 / torch.Tensor(reciprocal_weights))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset))

    train_loader = DataLoader(train_dataset, num_workers=args.num_worker, 
            batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(vad_dataset, num_workers=args.num_worker, 
            batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_worker, 
            batch_size=args.batch_size, shuffle=False)
    ood_loader = DataLoader(ood_dataset, num_workers=args.num_worker, 
            batch_size=args.batch_size, shuffle=False)
    ood_loader2 = DataLoader(ood_dataset2, num_workers=args.num_worker, 
            batch_size=args.batch_size, shuffle=False)        
            
            

    #################
    ## Train model ##
    #################
    
    model = PosteriorNetwork(N=N,
                             input_dims=input_dims,
                             output_dim=output_dim,
                             hidden_dims=hidden_dims,
                             kernel_dim=kernel_dim,
                             latent_dim=latent_dim,
                             architecture=architecture,
                             k_lipschitz=k_lipschitz,
                             no_density=no_density,
                             density_type=density_type,
                             n_density=n_density,
                             budget_function=budget_function,
                             batch_size=batch_size,
                             lr=lr,
                             loss=loss,
                             regr=regr,
                             seed=seed_model)
    
    full_config_dict = {'seed_dataset': seed_dataset,
                        'dataset_name': dataset_name,
                        'split': split,
                        'transform_min': transform_min,
                        'transform_max': transform_max,
                        'seed_model': seed_model,
                        'architecture': architecture,
                        # 'N': N,
                        'input_dims': input_dims,
                        'output_dim': output_dim,
                        'hidden_dims': hidden_dims,
                        'kernel_dim': kernel_dim,
                        'latent_dim': latent_dim,
                        'no_density': no_density,
                        'density_type': density_type,
                        'n_density': n_density,
                        'k_lipschitz': k_lipschitz,
                        'budget_function': budget_function,
                        'max_epochs': max_epochs,
                        'patience': patience,
                        'frequency': frequency,
                        'batch_size': batch_size,
                        'lr': lr,
                        'loss': loss,
                        'training_mode': training_mode,
                        'regr': regr}
    full_config_name = ''
    for k, v in full_config_dict.items():
        full_config_name += str(v) + '-'
    full_config_name = full_config_name[:-1]
    model_path = f'{directory_model}/model-dpn-{full_config_name}'

    train_losses, val_losses, train_accuracies, val_accuracies = [],[],[],[]
    
    ################
    ## Test model ##
    ################
    result_path = f'{directory_results}/results-dpn-{full_config_name}'
    checkpoint_path = f'{directory_model}/model-dpn-1-vanilla_dense_3-[0.75]-0.9-0.9-123-densenet-[1, 1, 1]-7-[64, 64, 64]-None-6-True-radial_flow-6-None-id-200-30-2-64-0.0001-CE-joint-1e-05'
    model.load_state_dict(torch.load(f'{checkpoint_path}')['model_state_dict'])
    print('Load model checkpoint from name succuessfully!')
    
    if args.test_only:
        metrics = test(model, train_loader, test_loader, ood_loader, result_path)
    else:
        #deterministic model: loss=CE, no_density=True
        feature_list = [128,output_dim] #currently set as 2
        metrics = test_baseline(model, output_dim, feature_list, train_loader, val_loader, test_loader, ood_loader, ood_loader2,args.name)
    
    

    results = {
        'model_path': model_path,
        'result_path': result_path,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }

    return {**results, **metrics}


results_metrics = run(# Dataset parameters
                        seed_dataset,  # Seed to shuffle dataset. int
                        directory_dataset,  # Path to dataset. string
                        dataset_name,  # Dataset name. string
                        ood_dataset_names,  # OOD dataset names.  list of strings
                        unscaled_ood,  # If true consider also unscaled versions of ood datasets. boolean
                        split,  # Split for train/val/test sets. list of floats
                        transform_min,  # Minimum value for rescaling input data. float
                        transform_max,  # Maximum value for rescaling input data. float

                        # Architecture parameters
                        seed_model,  # Seed to init model. int
                        directory_model,  # Path to save model. string
                        architecture,  # Encoder architecture name. string
                        input_dims,  # Input dimension. List of ints
                        output_dim,  # Output dimension. int
                        hidden_dims,  # Hidden dimensions. list of ints
                        kernel_dim,  # Input dimension. int
                        latent_dim,  # Latent dimension. int
                        no_density,  # Use density estimation or not. boolean
                        density_type,  # Density type. string
                        n_density,  # Number of density components. int
                        k_lipschitz,  # Lipschitz constant. float or None (if no lipschitz)
                        budget_function,  # Budget function name applied on class count. name

                        # Training parameters
                        directory_results,  # Path to save resutls. string
                        max_epochs,  # Maximum number of epochs for training
                        patience,  # Patience for early stopping. int
                        frequency,  # Frequency for early stopping test. int
                        batch_size,  # Batch size. int
                        lr,  # Learning rate. float
                        loss,  # Loss name. string
                        training_mode,  # 'joint' or 'sequential' training. string
                        regr)
for key in results_metrics:
    print(key,results_metrics[key])
