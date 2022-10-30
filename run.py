import logging
import os
import itertools
import argparse
import random
#from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, lr_scheduler
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomCrop
from torchvision.datasets import CIFAR10, SVHN

from utils import *
from image_dataloader import *
from PosteriorNetwork import PosteriorNetwork
from train import train, train_sequential
from test import test

# input argmuments
parser = argparse.ArgumentParser(description='Hybrid-EDL: improving EDL on imbalanced data')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.001,help='weight decay value')
parser.add_argument('--gpu_ids', default=[0], help='a list of gpus')
parser.add_argument('--num_worker', default=4, type=int, help='numbers of worker')
parser.add_argument('--batch_size', default=64, type=int, help='bacth size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--start_epochs', default=0, type=int, help='start epochs')
parser.add_argument('--class_number', default=10, type=int, help='number of classes')
parser.add_argument('--seed', default=1, type=int, help='seed for randome')
parser.add_argument('--data_dir', default='../data/CIFAR10', type=str, help='data directory')   
parser.add_argument('--ood_data_dir', default='../data/SVHN', type=str, help='data directory')    
parser.add_argument('--aug_scale', default=None, type=float, help='Augmentation multiplier')
parser.add_argument('--model_path',type=str, help='model saving directory')
parser.add_argument('--checkpoint', default=None, type=str, help='load checkpoint')
parser.add_argument('--test_only', default=False, type=bool, help='load checkpoint for testing')
parser.add_argument('--loss', default='UCE', type=str, help='loss function')
parser.add_argument('--latent_dim', default=6, type=int, help='latend dim')
parser.add_argument('--density_dim', default=6, type=int, help='density deep')
parser.add_argument('--no_density', default=False, type=bool, help='no flow')
parser.add_argument('--name', default='model', type=str, help='model save name')
parser.add_argument('--test_aug', default=None, type=str, help='with test augmentation type')
parser.add_argument('--test_aug_seed', default=None, type=int, help='with test augmentation')
args = parser.parse_args()

# Dataset parameters
seed_dataset= args.seed
directory_dataset=args.data_dir
dataset_name=args.name #not used indeed
ood_dataset_names='SVHN'

# Architecture parameters
seed_model=123
directory_model='./saved_models'
architecture='vgg' #update
input_dims=[32, 32, 3] #update 
output_dim=args.class_number #update
hidden_dims=[64,64] #update
kernel_dim=5
latent_dim=args.latent_dim #need to specify
no_density=args.no_density 
density_type='radial_flow'
n_density=args.density_dim
k_lipschitz=None
budget_function='id'

# Training parameters
directory_results='./saved_results'
max_epochs= 50  #200 10
patience= 5
frequency=2
batch_size=args.batch_size
lr=args.lr
loss=args.loss
training_mode='joint'
regr=1e-5

##not used 
unscaled_ood=True
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

    ##################
    ## Load dataset ##
    ##################
    dataset = CIFAR10(root='../data', train=True, download=True, transform=ToTensor())

    print('==> Preparing data..')
    mean, std = get_mean_and_std(dataset)
    print("MEAN",  mean, "STD", std)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = CIFAR10(root='../data', train=True, download=True, transform=transform_train)
   
    torch.manual_seed(0)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
 
    random.seed(seed_dataset)
    print('dataset_name:', dataset_name)
    if 'Balance' in dataset_name:
        sampling_ratio = [1]*10
    elif 'Light' in dataset_name:
        sampling_ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        random.shuffle(sampling_ratio)
    elif 'Mild' in dataset_name:
        sampling_ratio = [1, 0.89, 0.78,0.67,0.56,0.45,0.34,0.23,0.11,0.02]
        random.shuffle(sampling_ratio)
    elif 'Heavy' in dataset_name:
        sampling_ratio = [1,0.5,0.4,0.3,0.25,0.2,0.15,0.1,0.05,0.01]
        random.shuffle(sampling_ratio)
    print('data imbalance:', sampling_ratio )
  
    train_ds = imbalance_data(sampling_ratio, train_ds) #downsample training data
    val_ds = imbalance_data([1]*10, val_ds)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader =  torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2) #validation set
    

    test_dataset = CIFAR10(root='../data', train=False, transform=transform_test)  #test set
    test_dataset = imbalance_data([1]*10, test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    ood_dataset = SVHN(root='../data', split='test', download=True, transform=transform_test) ##OOD set
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(len(train_loader), len(val_loader), len(test_loader), len(ood_loader))

   

    #################
    ## Model ##
    #################
    #imbalance ration
    N = np.array([int(i*4500) for i in sampling_ratio])
    #N = np.array([int(i*4500) for i in [1]*10])
    N = torch.tensor(N)  
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
    print(full_config_dict)
    full_config_name = ''
    for k, v in full_config_dict.items():
        full_config_name += str(v) + '-'
    full_config_name = full_config_name[:-1]
    model_path = directory_model + '/model-dpn-'+full_config_name
    result_path = directory_results +  '/' + dataset_name
    print('save to:', result_path)
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)


    if not args.test_only:
        #Training the model, test with acc output
        train_losses, val_losses, train_accuracies, val_accuracies = train(model,
                                                                          train_loader,
                                                                          test_loader,
                                                                          val_loader,
                                                                          max_epochs=max_epochs,
                                                                          frequency=frequency,
                                                                          patience=patience,
                                                                          model_path=model_path,
                                                                          full_config_dict=full_config_dict)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        print('Succesussfully load model checkpoint from parameters')
        metrics = test(model, test_loader, val_loader, ood_loader, args.name, aug_type=args.test_aug, aug_seed=args.test_aug_seed)                                                           
   
    else:
        #Evaluate the model, 有问题
        train_losses, val_losses, train_accuracies, val_accuracies = [],[],[],[]
        # test_dataset = image_loader(args.class_number,args.data_dir, args.folds_file, args.test_fold, 
                        # False, "params_json", input_transform, stetho_id=args.stetho_id, aug_type=args.test_aug, aug_seed=args.test_aug_seed)
        # test_loader = DataLoader(test_dataset, num_workers=args.num_worker,  batch_size=args.batch_size, shuffle=False)
        
        ood_dataset = image_loader(args.class_number,args.ood_data_dir, args.ood_folds_file, [0], 
            False, "params_json", input_transform, stetho_id=args.stetho_id, aug_type=args.test_aug, aug_seed=args.test_aug_seed)
        ood_loader = DataLoader(ood_dataset, num_workers=args.num_worker, batch_size=args.batch_size, shuffle=False)
                    
        # ood_dataset2 = CIFAR10(root='../data', train=False,download=True, transform=input_transform)  #far OOD example
        # ood_loader2 = DataLoader(ood_dataset2, num_workers=args.num_worker, batch_size=args.batch_size, shuffle=False)  

        # checkpoint_path = directory_model+'/model-dpn-1-flow_dense_13-[0.75]-0.7-0.7-123-densenet-[1, 1, 1]-7-[64, 64]-None-6-False-radial_flow-12-None-id-50-30-1-64-0.0001-UCE-joint-1e-05' 
        # model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
        print('Load model checkpoint from name succuessfully!') 
        metrics = test(model, test_loader, val_loader, ood_loader, args.name, aug_type=args.test_aug, aug_seed=args.test_aug_seed)
    
    

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
