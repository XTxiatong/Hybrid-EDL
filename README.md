# Hybrid-EDL

This repository includes the code to reproduce our experiments in the paper entitled

*Hybrid-EDL: Improving Evidential Deep Learning for Uncertainty Quantification on Imbalanced Data*

presented in 2022 Trustworthy and Socially Responsible Machine Learning (TSRML 2022) co-located with NeurIPS 2022  



## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
conda activate hybrid-edl
conda env list
```



## Training & Evaluation

We use the experiments on CIFAR10 as the example to show how to run our codes. To train the model(s) in the paper, run 

```
sh run.sh
```

This is equivalent to 

```
python run.py --name cifar10_vgg_HeavyImbalance --seed 100 
```


### Trained Models

Trained models are saved in the folder `saved_models`. 

### Results

Model outputs are saved in the folder  `saved_results/cifar10_vgg_HeavyImbalance`.  Numerical results in Table 1 can be re-produced by  `python calibration_imbanlance.py`. 
The exampled output present the results when training the model with heavily imbalanced:

```
Before calibration:
acc: 0.6865
UAR: 0.6865
UAR majority 0.985
UAR minority: 0.6533333333333333
ECE: 0.17766500000000002
OOD ROAUC: 0.6816205016902275

After calibration:
acc: 0.7217
UAR: 0.7217
UAR majority 0.984
UAR minority: 0.6925555555555556
ECE: 0.16402499999999998
OOD ROAUC: 0.7010376805470191
```


*Note:  the folder  `src` is adatped from repo https://github.com/sharpenb/Posterior-Network.git

This is the vanilla EDL implementation we compared with.
