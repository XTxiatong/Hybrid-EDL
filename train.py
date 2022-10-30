import torch
import numpy as np
from sklearn import metrics

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print('========',device,'========')

def compute_loss_accuracy(model, loader, confuse=False):
    model.eval()
    with torch.no_grad():
        loss = 0
        for batch_index, (X, Y) in enumerate(loader):
            Y_hot = torch.zeros(Y.shape[0], loader.dataset.output_dim)
            Y = Y.reshape(Y.shape[0],1)
            Y_hot.scatter_(1, Y, 1)
            X, Y_hot = X.to(device), Y_hot.to(device)
            Y_pred = model(X, Y_hot)
            if batch_index == 0:
                Y_pred_all = Y_pred.view(-1).to("cpu")
                Y_all = Y.view(-1).to("cpu")
            else:
                Y_pred_all = torch.cat([Y_pred_all, Y_pred.view(-1).to("cpu")], dim=0)
                Y_all = torch.cat([Y_all, Y.view(-1).to("cpu")], dim=0)
            loss += model.grad_loss.item() #print loss, but not in backpropagation
        loss = loss / Y_pred_all.size(0)
        accuracy = ((Y_pred_all == Y_all).float().sum() / Y_pred_all.size(0)).item()
        conf_matrix = metrics.confusion_matrix(Y_all, Y_pred_all)
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
        uar = np.sum(conf_matrix.diagonal())/loader.dataset.output_dim
        if confuse:
            #print("Confusion Matrix")
            #print(conf_matrix)
            print('Acc:', accuracy, 'UAR:', uar)
    model.train()
    #print('loss:', loss)
    return loss, accuracy, uar


# Joint training for full model
def train(model, train_loader, test_loader, val_loader, max_epochs=200, frequency=2, patience=5, model_path='saved_model', full_config_dict={}):
    #print(model)
    model.to(device)
    model.train()
    train_losses, val_losses, train_accuracies, val_accuracies, val_uars = [], [], [], [], []
    best_val_loss = float("Inf")
    best_val_acc = 0

    for epoch in range(max_epochs):
        for batch_index, (X_train, Y_train) in enumerate(train_loader):
            #print('Y shape:', Y_train.shape[0], train_loader.dataset.output_dim )
            Y_train_hot = torch.zeros(Y_train.shape[0], train_loader.dataset.output_dim)
            Y_train = Y_train.reshape(Y_train.shape[0],1)
            Y_train_hot.scatter_(1, Y_train, 1)
            X_train, Y_train_hot = X_train.to(device), Y_train_hot.to(device)
            #print('batch:', batch_index)
            alpha = model(X_train, Y_train_hot,'alpha')
            model.step()
            
            # for name,p in model.named_parameters():
                # if 'density_estimation.0.transforms.0' in name:
                    # print(name, p.grad)

        if epoch % frequency == 0:
            # Stats on data sets
            # train_loss, train_accuracy = compute_loss_accuracy(model, train_loader)
            # train_losses.append(round(train_loss, 3))
            # train_accuracies.append(round(train_accuracy, 3))

            val_loss, val_accuracy, val_uar = compute_loss_accuracy(model, val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy) ##
            val_uars.append(val_uar)
            

            print("Epoch {} -> Val loss {} | Val Acc.: {}| Val UAR.: {}".format(epoch, round(val_losses[-1], 3), round(val_accuracies[-1], 3), round(val_uars[-1], 3)))
            
            print('Testing:')
            compute_loss_accuracy(model, test_loader, True)
            
            
            

            if val_losses[-1] < -1.:
                print("Unstable training")
                break
            if np.isnan(val_losses[-1]):
                print('Detected NaN Loss')
                break


            if best_val_acc < val_accuracies[-1]:
                best_val_acc = val_accuracies[-1]
                torch.save({'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': model.state_dict(), 'loss': best_val_loss}, model_path)
                print('Model saved by acc') 
                
                if best_val_acc > 0.8:
                    model.lr = 5e-5
                    print('change lr to 1e-5')
            if int(epoch / frequency) > patience and val_accuracies[-patience] >= max(val_accuracies[-patience:]):
                print('Early Stopping by acc.')
                break
                
            # if best_val_acc < val_uars[-1]:
                # best_val_acc = val_uars[-1]
                # torch.save({'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': model.state_dict(), 'loss': best_val_loss}, model_path)
                # print('Model saved by uar')

            # if int(epoch / frequency) > patience and val_uars[-patience] >= max(val_uars[-patience:]):
                # print('Early Stopping by uar.')
                # break



    return train_losses, val_losses, train_accuracies, val_accuracies


# Joint training method for ablated model
def train_sequential(model, train_loader, val_loader, max_epochs=200, frequency=2, patience=5, model_path='saved_model', full_config_dict={}):
    loss_1 = 'CE'
    loss_2 = model.loss

    print("### Encoder training ###")
    model.loss = loss_1
    model.no_density = True
    train_losses_1, val_losses_1, train_accuracies_1, val_accuracies_1 = train(model,
                                                                               train_loader,
                                                                               val_loader,
                                                                               max_epochs=max_epochs,
                                                                               frequency=frequency,
                                                                               patience=patience,
                                                                               model_path=model_path,
                                                                               full_config_dict=full_config_dict)
    print("### Normalizing Flow training ###")
    model.load_state_dict(torch.load(f'{model_path}')['model_state_dict'])
    for param in model.sequential.parameters():
        param.requires_grad = False
    model.loss = loss_2
    model.no_density = False
    train_losses_2, val_losses_2, train_accuracies_2, val_accuracies_2 = train(model,
                                                                               train_loader,
                                                                               val_loader,
                                                                               max_epochs=max_epochs,
                                                                               frequency=frequency,
                                                                               patience=patience,
                                                                               model_path=model_path,
                                                                               full_config_dict=full_config_dict)

    return train_losses_1 + train_losses_2, \
           val_losses_1 + val_losses_2, \
           train_accuracies_1 + train_accuracies_2, \
           val_accuracies_1 + val_accuracies_2
