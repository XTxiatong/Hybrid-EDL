# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:17:22 2022

@author: xiatong

Imbalance calibration
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import metrics
from math import *
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from scipy.stats import dirichlet
from collections  import Counter


def Brier(pro,label):
    Brier = 0
    for  i in range(len(pro)):
        label_one = np.zeros(max(label)+1)
        label_one[label[i]] =1
        pro_one = np.array(pro[i])
        brier =  sum((label_one-pro_one)**2)
        #print(brier)
        Brier+=brier
    return Brier/len(pro)

def nll(pro,label):
    nll = 0
    for  i in range(len(pro)):
        nll+=log(pro[i][label[i]])
    return -nll/len(pro)
        
        
def nor(x):
    return x/sum(x)

def entrop(x):
    en = 0
    for i in x:
        en += i*log(i)
    return -en

def z_score(x):
    xmin = min(x)
    xmax = max(x)
    return [(s-xmin)/(xmax-xmin) for s in x]

#for vanilla, alpha is the logit
path = 'cifar10_vgg_HeavyImbalance/'
Y_val = np.loadtxt(path+'Y_val_ID.txt',dtype=int)
alpha_val = np.loadtxt(path+'alpha_val_ID.txt')
Y = np.loadtxt(path+'Y_test_ID.txt',dtype=int)
alpha = np.loadtxt(path+'alpha_test_ID.txt')
prediction = np.argmax(alpha, axis=1)
pred_test = prediction
acc = metrics.accuracy_score(Y, prediction)
print('Before calibration:')
print('acc:', acc)
prob_test = np.array([nor(alpha[i,:]) for i in range(len(alpha))])
conf_matrix = metrics.confusion_matrix(Y, prediction)
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
print('UAR:', np.sum(conf_matrix.diagonal())/10)

"""
dataset_name: cifar10_vgg_HeavyImbalance
data imbalance: [0.25, 1, 0.2, 0.01, 0.3, 0.5, 0.15, 0.05, 0.1, 0.4]
[1125, 4500, 900, 45, 1350, 2250, 675, 225, 450, 1800]
print('data distribution')
"""

print('UAR majority',  conf_matrix.diagonal()[1])
print('UAR minority:', (np.sum(conf_matrix.diagonal()[2:]) + conf_matrix.diagonal()[0]) / 9)

#print(conf_matrix)
plt.figure(figsize=(4,3),dpi=300) 
sns.heatmap(conf_matrix,annot=True,fmt='.2f')



#9 is the majority
Y_binary = Y.copy()
prediction_binary = prediction.copy()
Y_binary[Y_binary!=0] = 1
Y_binary[Y_binary==0] = 0
prediction_binary[prediction_binary!=0] = 1
prediction_binary[prediction_binary==0] = 0
# Y_binary[Y_binary!=9] = 1
# Y_binary[Y_binary==9] = 0
# prediction_binary[prediction_binary!=9] = 1
# prediction_binary[prediction_binary==9] = 0
SE = sum(prediction_binary[Y_binary==1]==1) / sum(Y_binary==1)
SP = sum(prediction_binary[Y_binary==0]==0) / sum(Y_binary==0)
#print('Score:', (SE+SP)/2.0)
#print('SE:', SE)
#print('NLL:', nll(prob_test,Y))

############################################################################################
# Calibration of confidence 
nBin = 11 
conf_bars = np.linspace(0.5, 1, nBin)

def plot_acc_conf_bar(pro,prediction,y):
    acc_bars = []
    prop = []
    ECE = 0
    prop.append(0)
    acc_bars.append(0.5)
    for i, c in enumerate(conf_bars):
        if i == 0:
            continue 
        y_label = []
        y_pre = []
        for j in range(len(y)):
           con = max(pro[j,:])
           if con>conf_bars[i-1] and con<=c:
               y_label.append(y[j])
               y_pre.append(prediction[j])   
               
        conf_matrix = metrics.confusion_matrix(y_label, y_pre)
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
        #uar = np.sum(conf_matrix.diagonal())/10
        
        acc_bars.append(metrics.accuracy_score(y_label, y_pre))
        #acc_bars.append(uar)
        prop.append(len(y_label)*1.0/len(y))
        
    
    x = [i for i in range(nBin)]
    
    for i in range(1, nBin):
        ECE += prop[i]*np.abs(acc_bars[i]-conf_bars[i])
    print('ECE:', ECE)
    plt.figure(figsize=(8,4),dpi=300)
    plt.plot(x, acc_bars,'blue',label='ACC')
    plt.plot([0,10], [0.5,1],'r--',linewidth = '3',label='Optimal')
    plt.bar(x, prop,label='Proportion')  
    plt.grid()
    plt.tick_params(labelsize=15)
    plt.xticks(x, [str(c)[:4] for c in conf_bars])
    plt.legend(fontsize = 13)
    ax=plt.gca();#获得坐标轴的句柄
    #plt.title(name)
    plt.xlabel("Confidence",fontsize = 15)
    plt.ylabel("Accuracy",fontsize = 15)
    ax.spines['bottom'].set_linewidth(2) 
    ax.spines['left'].set_linewidth(2) 
    ax.spines['right'].set_linewidth(2) 
    ax.spines['top'].set_linewidth(2) 
    plt.show()   
    
plot_acc_conf_bar(prob_test,pred_test,Y) 

#################################################################################################
#  OOD
Y_ood2 = np.loadtxt(path+'Y_OOD.txt')
alpha_ood2 = np.loadtxt(path+'alpha_OOD.txt')
pred_ood2 = np.argmax(alpha_ood2, axis=1)
prob_ood2 = np.array([nor(alpha_ood2[i,:]) for i in range(len(alpha_ood2))])

in_entropy = [-dirichlet.entropy(alpha[i]) for i in range(len(alpha))]
ood_entropy2 = [-dirichlet.entropy(alpha_ood2[i]) for i in range(len(alpha_ood2))]


all_entropy2 = in_entropy + ood_entropy2
all_label2 = [1]*len(in_entropy) + [0]*len(ood_entropy2)
all_entropy2 = z_score(all_entropy2)
print('OOD ROAUC:', metrics.roc_auc_score(all_label2, all_entropy2))

##  Post Hoc Calibration
##  Based on which we search the best post-hoc calibration 
##  search a largest score in validation set
if False:
    W0 = 1
    W1 = 1
    W2 = 1
    W3 = 1
    W4 = 1
    W5 = 1
    W6 = 1
    W7 = 1
    W8 = 1
    W9 = 1
    
    UAR = 0
    for w0 in np.linspace(1,1,1):
        for w1 in np.linspace(1,1,1): ##Majority 
            for w2 in np.linspace(1,10,10):
                for w3 in np.linspace(1,50,20):
                    for w4 in np.linspace(1,1,1):
                        for w5 in np.linspace(1,1,1):
                            for w6 in np.linspace(1,1,1):
                                for w7 in np.linspace(1,50,20):
                                    for w8 in np.linspace(1,10,5):
                                        for w9 in np.linspace(1,1,1):
                                             
                                            alpha_new = alpha_val.copy()
                                            alpha_new[:,0] = alpha_val[:,0]*w0
                                            alpha_new[:,1] = alpha_val[:,1]*w1
                                            alpha_new[:,2] = alpha_val[:,2]*w2
                                            alpha_new[:,3] = alpha_val[:,3]*w3
                                            alpha_new[:,4] = alpha_val[:,4]*w4
                                            alpha_new[:,5] = alpha_val[:,5]*w5
                                            alpha_new[:,6] = alpha_val[:,6]*w6
                                            alpha_new[:,7] = alpha_val[:,7]*w7
                                            alpha_new[:,8] = alpha_val[:,8]*w8
                                            alpha_new[:,9] = alpha_val[:,9]*w9
                                            prediction_new = np.argmax(alpha_new, axis=1)
                                            conf_matrix = metrics.confusion_matrix(Y_val, prediction_new)
                                            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
                                            uar = np.sum(conf_matrix.diagonal())/10 
                                            #print(w0, w3, uar)
                                            if uar > UAR:
                                                UAR = uar
                                                W0 = w0
                                                W1 = w1
                                                W2 = w2
                                                W3 = w3
                                                W4 = w4
                                                W5 = w5
                                                W6 = w6
                                                W7 = w7
                                                W8 = w8
                                                W9 = w9
    print('Best Score:', UAR, W0,  W1, W2, W3, W4, W5, W6, W7, W8, W9)                
                    
print('-----------------------------------------------------------------')
print('After calibration:')
alpha_new = alpha.copy()  
alpha_ood2_new = alpha_ood2.copy()

#Saved serch results
if path == 'cifar10_vgg_HeavyImbalance/':      
    W0 = 1
    W1 = 1
    W2 = 4
    W3 = 1
    W4 = 1
    W5 = 1
    W6 = 1
    W7 = 29.4
    W8 = 7.75
    W9 = 1
 
    
alpha_new[:,0] = alpha_new[:,0]*W0
alpha_new[:,1] = alpha_new[:,1]*W1
alpha_new[:,2] = alpha_new[:,2]*W2
alpha_new[:,3] = alpha_new[:,3]*W3
alpha_new[:,4] = alpha_new[:,4]*W4
alpha_new[:,5] = alpha_new[:,5]*W5
alpha_new[:,6] = alpha_new[:,6]*W6
alpha_new[:,7] = alpha_new[:,7]*W7
alpha_new[:,8] = alpha_new[:,8]*W8
alpha_new[:,9] = alpha_new[:,9]*W9
alpha_ood2_new[:,0] = alpha_ood2_new[:,0]*W0
alpha_ood2_new[:,1] = alpha_ood2_new[:,1]*W1
alpha_ood2_new[:,2] = alpha_ood2_new[:,2]*W2
alpha_ood2_new[:,3] = alpha_ood2_new[:,3]*W3
alpha_ood2_new[:,4] = alpha_ood2_new[:,4]*W4
alpha_ood2_new[:,5] = alpha_ood2_new[:,5]*W5
alpha_ood2_new[:,6] = alpha_ood2_new[:,6]*W6
alpha_ood2_new[:,7] = alpha_ood2_new[:,7]*W7
alpha_ood2_new[:,8] = alpha_ood2_new[:,8]*W8
alpha_ood2_new[:,9] = alpha_ood2_new[:,9]*W9
    
prediction_new = np.argmax(alpha_new, axis=1)
acc = metrics.accuracy_score(Y, prediction_new)
print('acc:', acc)
conf_matrix = metrics.confusion_matrix(Y, prediction_new)
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
print('UAR:', np.sum(conf_matrix.diagonal())/10)

print('UAR majority',  conf_matrix.diagonal()[1])
print('UAR minority:', (np.sum(conf_matrix.diagonal()[2:]) + conf_matrix.diagonal()[0]) / 9)

plt.figure(figsize=(4,3),dpi=300) 
sns.heatmap(conf_matrix,annot=True,fmt='.2f')

prediction_binary_new = prediction_new.copy()
prediction_binary_new[prediction_binary_new!=0] = 1
prediction_binary_new[prediction_binary_new==0] = 0
# prediction_binary_new[prediction_binary_new!=9] = 1
# prediction_binary_new[prediction_binary_new==9] = 0


SE = sum(prediction_binary[Y_binary==1]==1) / sum(Y_binary==1)
SP = sum(prediction_binary[Y_binary==0]==0) / sum(Y_binary==0)
#print('Score:', (SE+SP)/2.0)
# print('SE:', SE)
# print('NLL:', nll(prob_test,Y))
# print('Brier', Brier(prob_test,Y))
prob_new_test = np.array([nor(alpha_new[i,:]) for i in range(len(alpha_new))])
plot_acc_conf_bar(prob_new_test,prediction_new,Y)


in_entropy_new = [-dirichlet.entropy(alpha_new[i]) for i in range(len(alpha_new))]
ood_entropy2_new = [-dirichlet.entropy(alpha_ood2_new[i]) for i in range(len(alpha_ood2_new))]


all_entropy2 = in_entropy_new + ood_entropy2_new
all_label2 = [1]*len(in_entropy_new) + [0]*len(ood_entropy2_new)
all_entropy2 = z_score(all_entropy2)
print('OOD ROAUC:', metrics.roc_auc_score(all_label2, all_entropy2))


