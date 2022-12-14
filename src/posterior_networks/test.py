import torch
import pickle
from src.results_manager.metrics_prior import accuracy, confidence, brier_score, anomaly_detection
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def compute_X_Y_alpha(model, loader, alpha_only=False):
    for batch_index, (X, Y) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        alpha_pred = model(X, None, return_output='alpha', compute_loss=False)
        if batch_index == 0:
            X_duplicate_all = X.to("cpu")
            orig_Y_all = Y.to("cpu")
            alpha_pred_all = alpha_pred.to("cpu")
        else:
            X_duplicate_all = torch.cat([X_duplicate_all, X.to("cpu")], dim=0)
            orig_Y_all = torch.cat([orig_Y_all, Y.to("cpu")], dim=0)
            alpha_pred_all = torch.cat([alpha_pred_all, alpha_pred.to("cpu")], dim=0)
    if alpha_only:
        return alpha_pred_all
    else:
        return orig_Y_all, X_duplicate_all, alpha_pred_all


def test(model, test_loader, ood_dataset_loaders, result_path='saved_results'):
    model.to(device)
    model.eval()

    with torch.no_grad():
        orig_Y_all, X_duplicate_all, alpha_pred_all = compute_X_Y_alpha(model, test_loader)
        np.savetxt('alpha_test_ID.txt', alpha_pred_all.cpu().detach().numpy())
        np.savetxt('X_test_ID.txt', X_duplicate_all.cpu().detach().numpy())
        np.savetxt('Y_test_ID.txt', orig_Y_all.cpu().detach().numpy())
        
        # print('==y==')
        # print(orig_Y_all.cpu().detach().numpy())
        # corrects = (orig_Y_all.squeeze() == alpha_pred_all.max(-1)[1]).cpu().detach().numpy()
        # print('==correct==')
        # print(corrects)
        # print('==alpha==')
        # print(alpha_pred_all.cpu().detach().numpy())
        # print('==p==')
        # p = torch.nn.functional.normalize(alpha_pred_all, p=1, dim=-1)
        # scores = p.max(-1)[0].cpu().detach().numpy()
        # print(scores)
        
        
        # Save each data result
        n_test_samples = orig_Y_all.size(0)
        full_results_dict = {'Y': orig_Y_all.cpu().detach().numpy(),
                             'X': X_duplicate_all.view(n_test_samples, -1).cpu().detach().numpy(),
                             'alpha': alpha_pred_all.cpu().detach().numpy()}
        with open(f'{result_path}.pickle', 'wb') as handle:
            pickle.dump(full_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metrics
        metrics = {}
        metrics['accuracy'] = accuracy(Y=orig_Y_all, alpha=alpha_pred_all)
        metrics['confidence_aleatoric'] = confidence(Y= orig_Y_all, alpha=alpha_pred_all, score_type='APR', uncertainty_type='aleatoric')
        metrics['confidence_epistemic'] = confidence(Y= orig_Y_all, alpha=alpha_pred_all, score_type='APR', uncertainty_type='epistemic')
        metrics['brier_score'] = brier_score(Y= orig_Y_all, alpha=alpha_pred_all)
        for ood_dataset_name, ood_loader in ood_dataset_loaders.items():
            Y,X,ood_alpha_pred_all = compute_X_Y_alpha(model, ood_loader)
            print(ood_alpha_pred_all.size(),alpha_pred_all.size())
            print('==alpha==')
            print(ood_alpha_pred_all.cpu().detach().numpy())
            np.savetxt('alpha_OOD.txt', ood_alpha_pred_all.cpu().detach().numpy())
            np.savetxt('X_OOD.txt', X.cpu().detach().numpy())
            np.savetxt('Y_OOD.txt', Y.cpu().detach().numpy())
            metrics[f'anomaly_detection_aleatoric_{ood_dataset_name}'] = anomaly_detection(alpha=alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR', uncertainty_type='aleatoric')
            metrics[f'anomaly_detection_epistemic_{ood_dataset_name}'] = anomaly_detection(alpha=alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR', uncertainty_type='epistemic')

    return metrics


