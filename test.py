import torch
import pickle
from src.results_manager.metrics_prior import accuracy, confidence, brier_score, anomaly_detection
import numpy as np
from utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


from sklearn import metrics

def get_score(Y, alpha):
    predicted = np.argmax(alpha.cpu().detach().numpy(), axis=1)
    Y_test = Y.cpu().detach().numpy()
    
    acc = metrics.accuracy_score(Y_test, predicted)
    conf_matrix = metrics.confusion_matrix(Y_test, predicted)
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
    print("Confusion Matrix", conf_matrix)
    print('UAR:', np.sum(conf_matrix.diagonal())/len(set(Y_test)))
    print("acc", acc)
    
    score = 0
    sp = 0
    se = 0
    for j in range(len(Y_test)):
        if Y_test[j]==0 and predicted[j]==0:
            score += 1
            sp += 1
        if Y_test[j]>0 and predicted[j]>0:
            score += 1
            se += 1
    print("Score", score/len(Y_test))
    print("Se", se/sum(Y_test>0))
    print("Sp", sp/sum(Y_test==0))    


def compute_X_Y_alpha(model, loader, alpha_only=False):
    for batch_index, (X, Y) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        alpha_pred = model(X, None, return_output='alpha', compute_loss=False)
        emebd = model(X, None, return_output='latent', compute_loss=False)
        if batch_index == 0:
            X_duplicate_all = X.to("cpu")
            #print('bacth 0:', X_duplicate_all.shape)
            orig_Y_all = Y.to("cpu")
            alpha_pred_all = alpha_pred.to("cpu")
            emebd_all = emebd.to('cpu')
        else:
            X_duplicate_all = torch.cat([X_duplicate_all, X.to("cpu")], dim=0)
            orig_Y_all = torch.cat([orig_Y_all, Y.to("cpu")], dim=0)
            alpha_pred_all = torch.cat([alpha_pred_all, alpha_pred.to("cpu")], dim=0)
            emebd_all = torch.cat([emebd_all,emebd.to('cpu')],dim=0)
    if alpha_only:
        return alpha_pred_all
    else:
        return orig_Y_all, X_duplicate_all, alpha_pred_all, emebd_all


def test(model, test_loader, val_loader, ood_dataset_loaders, result_path='saved_results', aug_type=None, aug_seed=None):
    path = 'saved_results/' + result_path + '/'
    name = '' if aug_type == None else aug_type + '_' + str(aug_seed) + '_'
    model.to(device)
    model.eval()
    
    metrics = {}
    with torch.no_grad():
        orig_Y_all, X_duplicate_all, alpha_pred_all,emebd_all = compute_X_Y_alpha(model, val_loader)
        np.savetxt(path+name+'alpha_val_ID.txt', alpha_pred_all.cpu().detach().numpy())
        np.savetxt(path+name+'Y_val_ID.txt', orig_Y_all.cpu().detach().numpy())
       
        print('===validation set===')
        get_score(Y=orig_Y_all, alpha=alpha_pred_all)
        
        orig_Y_all, X_duplicate_all, alpha_pred_all,emebd_all = compute_X_Y_alpha(model, test_loader)
        np.savetxt(path+name+'alpha_test_ID.txt', alpha_pred_all.cpu().detach().numpy())
        np.savetxt(path+name+'Y_test_ID.txt', orig_Y_all.cpu().detach().numpy())
        metrics['accuracy'] = accuracy(Y=orig_Y_all, alpha=alpha_pred_all)
        metrics['confidence_aleatoric'] = confidence(Y= orig_Y_all, alpha=alpha_pred_all, score_type='APR', uncertainty_type='aleatoric')
        metrics['confidence_epistemic'] = confidence(Y= orig_Y_all, alpha=alpha_pred_all, score_type='APR', uncertainty_type='epistemic')
        metrics['brier_score'] = brier_score(Y= orig_Y_all, alpha=alpha_pred_all)
        print('===In-distribution test set===')
        get_score(Y=orig_Y_all, alpha=alpha_pred_all)
        
        Y,X,ood_alpha_pred_all,emebd_all = compute_X_Y_alpha(model, ood_dataset_loaders)
        np.savetxt(path+name+'alpha_OOD.txt', ood_alpha_pred_all.cpu().detach().numpy())
        np.savetxt(path+name+'Y_OOD.txt', Y.cpu().detach().numpy())
        ood_dataset_name = 'ood_dataset'
        metrics[f'anomaly_detection_aleatoric_{ood_dataset_name}'] = anomaly_detection(alpha=alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR', uncertainty_type='aleatoric')
        metrics[f'anomaly_detection_epistemic_{ood_dataset_name}'] = anomaly_detection(alpha=alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR', uncertainty_type='epistemic')
        print('===Out-of-distribution test set 1===')
        get_score(Y=Y, alpha=ood_alpha_pred_all)
        
        # Y,X,ood_alpha_pred_all,emebd_all = compute_X_Y_alpha(model, ood_dataset_loaders2)
        # np.savetxt(path+name+'alpha_OOD2.txt', ood_alpha_pred_all.cpu().detach().numpy())
        # np.savetxt(path+name+'Y_OOD2.txt', Y.cpu().detach().numpy())
        # ood_dataset_name = 'ood_dataset2'
        # metrics[f'anomaly_detection_aleatoric_{ood_dataset_name}'] = anomaly_detection(alpha=alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR', uncertainty_type='aleatoric')
        # metrics[f'anomaly_detection_epistemic_{ood_dataset_name}'] = anomaly_detection(alpha=alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR', uncertainty_type='epistemic')
        # print('===Out-of-distribution test set 2===')

    return metrics



def test_baseline(model, output_dim, feature_list, train_loader, val_loader, test_loader, ood_loader, ood_loader2, result_path='saved_results'):
    model.to(device)
    model.eval()
    path = 'saved_results/' + result_path + '/'
    name = ''
    with torch.no_grad():
        sample_mean, sample_covariance = sample_estimator(model, output_dim, feature_list, train_loader)
        #print('training distribution:', sample_mean, sample_covariance)
        
        # sample_mean_val, sample_covariance_val = sample_estimator(model, output_dim, feature_list, val_loader)
        # print('validation distribution:', sample_mean_val, sample_covariance_val)
        
        magnitude = 0.0005 #noise level
        for i in range(len(feature_list)): #layer-wise score
            M_val = get_Mahalanobis_score(model, val_loader, output_dim, path, True, sample_mean, sample_covariance, i, magnitude)
            M_val = np.asarray(M_val, dtype=np.float32)
            if i == 0:
                Mahalanobis_val = M_val.reshape((M_val.shape[0], -1))
            else:
                Mahalanobis_val = np.concatenate((Mahalanobis_val, M_val.reshape((M_val.shape[0], -1))), axis=1)
        print(Mahalanobis_val) 
        print('mean of validaiotn set:', np.mean(Mahalanobis_val,axis=0)) 
        
        
        for i in range(len(feature_list)): #layer-wise score
            M_in = get_Mahalanobis_score(model, test_loader, output_dim, path, False, sample_mean, sample_covariance, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
        print(Mahalanobis_in) 
        print('mean of test set:', np.mean(Mahalanobis_in,axis=0))    
        
        for i in range(len(feature_list)): #layer-wise score
            M_out = get_Mahalanobis_score(model, ood_loader, output_dim, path, False, sample_mean, sample_covariance, i, magnitude)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
        print(Mahalanobis_out)  
        print('mean of OOD:', np.mean(Mahalanobis_out,axis=0))   

        for i in range(len(feature_list)): #layer-wise score
            M_out2 = get_Mahalanobis_score(model, ood_loader2, output_dim, path, False, sample_mean, sample_covariance, i, magnitude)
            M_out2 = np.asarray(M_out2, dtype=np.float32)
            if i == 0:
                Mahalanobis_out2 = M_out2.reshape((M_out2.shape[0], -1))
            else:
                Mahalanobis_out2 = np.concatenate((Mahalanobis_out2, M_out2.reshape((M_out2.shape[0], -1))), axis=1)
        print(Mahalanobis_out2)  
        print('mean of OOD:', np.mean(Mahalanobis_out2,axis=0))                
        #Y,X,ood_alpha_pred_all,emebd_all = compute_X_Y_alpha(model, ood_dataset_loaders)
    
    np.savetxt(path+'Mahalanobis_val.txt', Mahalanobis_val)
    np.savetxt(path+'Mahalanobis_ID.txt', Mahalanobis_in)
    np.savetxt(path+'Mahalanobis_OOD.txt', Mahalanobis_out)
    np.savetxt(path+'Mahalanobis_OOD2.txt', Mahalanobis_out2)
    metrics = {}
    return metrics

