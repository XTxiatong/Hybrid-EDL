import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.distributions.dirichlet import Dirichlet
from src.architectures.linear_sequential import linear_sequential
from src.architectures.convolution_linear_sequential import convolution_linear_sequential
from src.architectures.vgg_sequential import vgg16_bn
from src.architectures.resnet_sequential import resnet18
from torchvision.models import resnet18, resnet34, resnet50, densenet121, vgg16
from src.architectures.alexnet_sequential import alexnet
from src.posterior_networks.NormalizingFlowDensity import NormalizingFlowDensity
from src.posterior_networks.BatchedNormalizingFlowDensity import BatchedNormalizingFlowDensity
from src.posterior_networks.MixtureDensity import MixtureDensity

__budget_functions__ = {'one': lambda N: torch.ones_like(N),
                        'log': lambda N: torch.log(N + 1.),
                        'id': lambda N: N,
                        'id_normalized': lambda N: N / N.sum(),
                        'exp': lambda N: torch.exp(N),
                        'parametrized': lambda N: torch.nn.Parameter(torch.ones_like(N).to(N.device))}


class PosteriorNetwork(nn.Module):
    def __init__(self, N,  # Count of data from each class in training set. list of ints
                 input_dims,  # Input dimension. list of ints
                 output_dim,  # Output dimension. int
                 hidden_dims=[64,64,64],  # Hidden dimensions. list of ints, changed 
                 kernel_dim=None,  # Kernel dimension if conv architecture. int
                 latent_dim=6,  # Latent dimension. int
                 architecture='linear',  # Encoder architecture name. int
                 k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
                 no_density=False,  # Use density estimation or not. boolean
                 density_type='radial_flow',  # Density type. string
                 n_density=8,  # Number of density components. int
                 budget_function='id',  # Budget function name applied on class count. name
                 batch_size=64,  # Batch size. int
                 lr=1e-3,  # Learning rate. float
                 loss='UCE',  # Loss name. string
                 regr=1e-5,  # Regularization factor in Bayesian loss. float
                 seed=0,
                 drop_prob=0.5):  # Random seed for init. int
        super().__init__()

        torch.cuda.manual_seed(seed)
        #torch.set_default_tensor_type(torch.DoubleTensor)

        # Architecture parameters
        self.input_dims, self.output_dim, self.hidden_dims, self.kernel_dim, self.latent_dim = input_dims, output_dim, hidden_dims, kernel_dim, latent_dim
        self.k_lipschitz = k_lipschitz
        self.no_density, self.density_type, self.n_density = no_density, density_type, n_density
     
        if budget_function in __budget_functions__:
            self.N, self.budget_function = __budget_functions__[budget_function](N), budget_function
        else:
            raise NotImplementedError
            
        print(self.N)
        # Training parameters
        self.batch_size, self.lr = batch_size, lr
        self.loss, self.regr = loss, regr

        # Encoder -- Feature selection
        if architecture == 'linear':
            self.sequential = linear_sequential(input_dims=self.input_dims,
                                                hidden_dims=self.hidden_dims,
                                                output_dim=self.latent_dim,
                                                k_lipschitz=self.k_lipschitz)
        elif architecture == 'conv':
            assert len(input_dims) == 3
            self.sequential = convolution_linear_sequential(input_dims=self.input_dims,
                                                            linear_hidden_dims=self.hidden_dims,
                                                            conv_hidden_dims=[64, 64, 64],
                                                            output_dim=self.latent_dim,
                                                            kernel_dim=self.kernel_dim,
                                                            k_lipschitz=self.k_lipschitz)
            self.kl_linear =  linear_sequential(input_dims=latent_dim,
                                                hidden_dims=[],
                                                output_dim=self.latent_dim,
                                                k_lipschitz=self.k_lipschitz) 
            
            
        elif architecture == 'vgg':
            self.sequential = nn.Sequential(vgg16(pretrained=True),nn.Dropout(drop_prob), nn.Linear(1000, 128), nn.ReLU(True)) ## this the 128-d embeding
            self.kl_linear =  linear_sequential(input_dims=128,
                                                hidden_dims=self.hidden_dims,
                                                output_dim=self.latent_dim,
                                                k_lipschitz=self.k_lipschitz) 
            
            
        elif architecture == 'resnet':           
            self.sequential = resnet50(pretrained=True) # 128
            #print(self.sequential)
            num_ftrs = self.sequential.fc.in_features #replacing fc layer with sequential.fc
            self.sequential.fc = nn.Sequential(nn.Dropout(drop_prob), nn.Linear(num_ftrs, 128), nn.ReLU(True), 
                nn.Dropout(drop_prob), nn.Linear(128, 128), nn.ReLU(True)) ## this the 128-d embeding
            self.kl_linear =  linear_sequential(input_dims=128,
                                                hidden_dims=self.hidden_dims,
                                                output_dim=self.latent_dim,
                                                k_lipschitz=self.k_lipschitz) 
        elif architecture == 'densenet':
            self.sequential = densenet121(pretrained=True) # 128
            #print(self.sequential)
            num_ftrs = self.sequential.classifier.in_features #replacing fc layer with sequential.fc
            self.sequential.classifier = nn.Sequential(nn.Dropout(drop_prob), nn.Linear(num_ftrs, 128), nn.ReLU(True), 
                nn.Dropout(drop_prob), nn.Linear(128, 128), nn.ReLU(True)) ## this the 128-d embeding
            self.kl_linear =  linear_sequential(input_dims=128,
                                                hidden_dims=self.hidden_dims,
                                                output_dim=self.latent_dim,
                                                k_lipschitz=self.k_lipschitz) 
        else:
            raise NotImplementedError
            
        self.batch_norm = nn.BatchNorm1d(num_features=self.latent_dim)
        self.cls_fc = nn.Linear(128, self.output_dim)
        self.projection = nn.Sequential(nn.Linear(128, 64, bias=False), nn.BatchNorm1d(64),
                               nn.ReLU(inplace=True), nn.Linear(64, 32, bias=True))

        # Normalizing Flow -- Normalized density on latent space
        if self.density_type == 'planar_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type) for c in range(self.output_dim)])
        elif self.density_type == 'radial_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type) for c in range(self.output_dim)])
        elif self.density_type == 'batched_radial_flow':
            self.density_estimation = BatchedNormalizingFlowDensity(c=self.output_dim, dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type.replace('batched_', ''))
        elif self.density_type == 'iaf_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type) for c in range(self.output_dim)])
        elif self.density_type == 'normal_mixture':
            self.density_estimation = nn.ModuleList([MixtureDensity(dim=self.latent_dim, n_components=n_density, mixture_type=self.density_type) for c in range(self.output_dim)])
        else:
            raise NotImplementedError
        self.softmax = nn.Softmax(dim=-1)

        # Optimizer
        ignored_params = list(map(id, self.density_estimation.parameters())) 
        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters()) 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, input, label, return_output='hard', compute_loss=True):
        batch_size = input.size(0)

        if self.N.device != input.device:
            self.N = self.N.to(input.device)

        if self.budget_function == 'parametrized':
            N = self.N / self.N.sum()
        else:
            N = self.N

        # Forward
        zk = self.sequential(input)  #128 --> to linear classifer/protector/normflow
        
        if self.no_density:  # Ablated model without density estimation
            #print('No density!!!!!!!!!')
            logits = self.cls_fc(zk)
            alpha = torch.exp(logits)
            prob_pred = self.softmax(logits)
            #print('prob_pred',prob_pred)
        else:  # Full model with density estimation
            #print('With density!!!!!!!!!')
            zk2 = self.kl_linear(zk)
            zk2 = self.batch_norm(zk2)
            log_q_zk = torch.zeros((batch_size, self.output_dim)).to(zk2.device.type)
            alpha = torch.zeros((batch_size, self.output_dim)).to(zk2.device.type)

            if isinstance(self.density_estimation, nn.ModuleList):
                for c in range(self.output_dim):
                    log_p = self.density_estimation[c].log_prob(zk2) #This should not be large-negative 
                    log_q_zk[:, c] = log_p
                    #print('log p:', log_p)
                    alpha[:, c] = 1. + (N[c] * torch.exp(log_q_zk[:, c]))
            else:
                log_q_zk = self.density_estimation.log_prob(zk2)
                alpha = 1. + (N[:, None] * torch.exp(log_q_zk)).permute(1, 0)

            pass

            prob_pred = torch.nn.functional.normalize(alpha, p=1)
        output_pred = self.predict(prob_pred)
       
        

        # Loss
        if compute_loss:
            if self.loss == 'CE':
                self.grad_loss = self.CE_loss(prob_pred, label)
            elif self.loss == 'UCE':
                self.grad_loss = self.UCE_loss(alpha, label)
            elif self.loss == 'CL':
                pass ##defined outsided
            else:
                raise NotImplementedError

        if return_output == 'hard':
            return output_pred
        elif return_output == 'soft':
            return prob_pred
        elif return_output == 'alpha':
            return alpha
        elif return_output == 'latent':
            return zk
        elif self.loss == 'CL' and return_output == 'projection':
            out_proj = self.projection(zk)  
            return out_proj
        else:
            raise AssertionError

    def CE_loss(self, prob_pred, label):
        with autograd.detect_anomaly():
            CE_loss = - torch.sum(label.squeeze() * torch.log(prob_pred))
            return CE_loss

    def UCE_loss(self, alpha, label):
        with autograd.detect_anomaly():
            alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.output_dim)
            entropy_reg = Dirichlet(alpha).entropy()
            UCE_loss = torch.sum(label * (torch.digamma(alpha_0) - torch.digamma(alpha))) - self.regr * torch.sum(entropy_reg)
            #print('UCE_loss:',label, alpha, UCE_loss)
            return UCE_loss
           

    def step(self):
        self.optimizer.zero_grad()
        self.grad_loss.backward()
        self.optimizer.step()
      

    def predict(self, p):
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred
        
    def feature_list(self,input):
        #only used for deterministic model
        if self.N.device != input.device:
            self.N = self.N.to(input.device)
        out_list = []
        out = self.sequential(input)
        out_list.append(out)
        # out = self.sequential(out)
        # out_list.append(out)
        out = self.cls_fc(out)
        out_list.append(out)
        return out_list
        
    # function to extact a specific feature
    def intermediate_forward(self, input, layer_index):
        #only used for deterministic model
        if self.N.device != input.device:
            self.N = self.N.to(input.device) 
        if layer_index == 0:
            out = self.sequential(input)
        elif layer_index == 1:
            out = self.sequential(input)
            out = self.cls_fc(out)
        return out        