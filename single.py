#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import os
import torch
import itertools


import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

from scipy.signal import savgol_filter
from captum.attr import IntegratedGradients




# In[2]:


bio = 'Saliva' #Define which biofluid to use
technique = 'FTIR' #Define which spectroscopy technique to use


# In[3]:


# Auxiliar functions
def baseline_poly(x, y, order):
    coefs = poly.polyfit(x, y, order)
    ffit = poly.polyval(x, coefs)
    return y - ffit


def read_biofluids(path, form, quick_labels, biofluids, n_samples,
                            cut, rel, laser_wl,technique):
    X = {}
    lmbs = {}
    
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            filepath = subdir + os.sep + filename
            print(filepath)
            if filepath.endswith(".txt"):
                if form.lower() in filename.lower():
                    f_split = filename.split('_')
                    ID = f_split[0]
                    group = f_split[1]
    
                    if group in biofluids:
                        try:
                            aux = np.loadtxt(filepath, delimiter=',', unpack=False)
                            num = len(aux)//n_samples # aux.shape = (5120, 2), 2 because the x and y dims, 5120 bcs 1024 for measure and 5 measures
                            aux = aux[:num*n_samples] # Remove tail errors
  
                        except:
                            print('Error loading')

                        for g in range(len(aux)//n_samples):
                            if technique == 'FTIR':
                                aux2 = aux[g*n_samples:(g+1)*n_samples]
                                sorted_array = aux2[np.argsort(aux2[:, 0])]

                                # Baseline correction (ALS/Polyfit)
                                corrected = baseline_poly(sorted_array[:,0], sorted_array[:,1], 5)
                                corrected = sorted_array[:,1]

                                # Smoothing (Savitzky Golay)
                                smoothed = savgol_filter(corrected, window_length=7, polyorder=3)
#                                 smoothed = savgol_filter(corrected, window_length=201, polyorder=1)
#                                 print(smoothes)
                                smoothed = corrected

                                # Normalization (SNV)
                                snv = (smoothed - np.mean(smoothed)) / np.std(smoothed) #SNV

                                # Obtain derivatives
                                first_derv = snv[8:] - snv[:-8]
                                second_derv = first_derv[8:] - first_derv[:-8]

                                shorter = np.min((len(snv),len(first_derv), len(second_derv)))
                                X[(int(ID), group, g)] = [snv[:shorter], first_derv[:shorter], second_derv[:shorter]]
                                lmbs[(int(ID), group, g)] = aux2[:, 0]
                            else:
                                aux2 = aux[g*n_samples + cut:(g+1)*n_samples]
                                aux2[:, 0] = rel/laser_wl - rel/aux2[:, 0] # From wavelenght (nm) to Raman shift (cm⁻1)
                                sorted_array = aux2[np.argsort(aux2[:, 0])]

                                if np.min(sorted_array[:,0]) > 380:
                                    # Baseline correction (ALS/Polyfit)
                                    corrected = baseline_poly(sorted_array[:,0], sorted_array[:,1], 5)
                                    corrected = sorted_array[:,1]

                                    # Smoothing (Savitzky Golay)
                                    smoothed = savgol_filter(corrected, window_length=7, polyorder=3)
                                    smoothed = corrected

                                    # Normalization (SNV)
                                    snv = (smoothed - np.mean(smoothed)) / np.std(smoothed) #SNV

                                    # Obtain derivatives
                                    first_derv = snv[8:] - snv[:-8]
                                    second_derv = first_derv[8:] - first_derv[:-8]

                                    shorter = np.min((len(snv),len(first_derv), len(second_derv)))
                                    X[(int(ID), group, g)] = [snv[:shorter], first_derv[:shorter], second_derv[:shorter]]
                                    lmbs[(int(ID), group, g)] = aux2[:, 0]
                                

    return X, lmbs


# In[4]:


# Load dataset

if technique == 'FTIR':
    
    laser_wl = 785.15 # Laser wavelenght in nm
    rel = 1e7
    cut = 178
    
    path = 'FTIR_DB'
    n_samples = 846 
else:
    path = 'Raman_DB'
    n_samples = 1024
    laser_wl = 785.15 # Laser wavelenght in nm
    rel = 1e7
    cut = 178 # Remove the first 178 samples to reduce laser noise in near wavelengths -> from 810.59 nm to 909.88 nm
     
biofluids = ['plasma', 'saliva']
form = 'Dry' # 'Wet' or 'Dry'

info = pd.read_excel('Biofluid_list_annotated_v2.xlsx', index_col=(0))
quick_labels = info.loc[:, 'Label'].to_dict()

X = {}

wv_aux = []


X, lmbs = read_biofluids(path, form, quick_labels, biofluids, n_samples∏ cut, rel, laser_wl,technique)


# In[5]:


#%% Get the matched patients for each dataset
k = np.reshape(list(X.keys()), (-1, 3))
patients_plasma = set(k[k[:, 1]=='plasma'][:, 0])
patients_saliva = set(k[k[:, 1]=='saliva'][:, 0])

patients_matches = list(patients_plasma  & patients_saliva)
patients_matches = set([int(i) for i in patients_matches])


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


#%% K-SPLIT the dataset
import random
labels_pos = info[info['Label'] == 1].index.values
labels_neg = info[info['Label'] == 0].index.values

labels_pos = list(set(labels_pos) & patients_matches)
labels_neg = list(set(labels_neg) & patients_matches)



seed = 14
random.Random(seed).shuffle(labels_pos)
random.Random(seed).shuffle(labels_neg)

K = len(labels_neg)
# K = 5
labels_pos = np.array_split(labels_pos, K)
labels_neg = np.array_split(labels_neg, K)


# In[8]:


# Imports
import torch

from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.metrics import accuracy_score

# from sklearn.preprocessing import StandardScaler


# In[31]:


# Auxiliar functions

def extract_X_single_staging(X, case, group, matching_labels, quick_labels, test_labels, info):
    '''
    Obtain X but with the mean of all of combination of the samples lenght
    Used in train to remove noise and increase the dataset size
    '''
    
    labels_case = info[info['Label'] == case].index.values
    
    labels = list(set(labels_case) & matching_labels)

    X_train = []
    y_train = []
    
    X_test = []
    y_test = []
    
    for l in labels:
        xx3 = []
        stage = info[info.index == l].Staging.values[0]
        
        if stage == 0:
            stage = [1, 0, 0]
        elif stage == 1 or stage == 2:
            stage = [0, 1, 0]
        elif stage == 3 or stage == 4:
            stage = [0, 0, 1] 
        else:
            print("Error during one-hot enconding")
        xx2 = X.get((l, group.lower(),0), [])
#         print(f'found {group}:{l}')
        xx3 = np.array(xx2)
        if l in test_labels:
            X_test.append(xx3)
            y_test.append(stage)
        else:
            X_train.append(xx3)
            y_train.append(stage)
 

    return X_train, y_train, X_test, y_test


def obtain_features_importance(model, inputs, target):
    attrs = []
    ig = IntegratedGradients(model.to('cpu'))
    test_input_tensor = torch.from_numpy(inputs).type(torch.FloatTensor).to('cpu')
    target_tensor = torch.from_numpy(target).type(torch.FloatTensor)
    
    for inp, targ in zip(test_input_tensor, target_tensor):
        inp = np.reshape(inp, (1, 3, -1))
        if targ[0] == 1:
            t = 0
        elif targ[1] == 1:
            t = 1
        elif targ[2] == 1:
            t = 2
        else:
            print('ERROR')
        inp.requires_grad_()
        attributions, approximation_error = ig.attribute(inp, target=t,
                                                          return_convergence_delta=True)
        attrs.append(attributions.detach().numpy())
    
    attributions = np.mean(attrs, 0)[0]
    fi_bio_deriv = np.mean(attributions, -1)

    attributions = np.mean(attributions, 0)

    return attributions, fi_bio_deriv


# In[32]:


class DCNN_one_staging(nn.Module):
    def __init__(self, in_size):
        super(DCNN_one_staging, self).__init__()
        self.in_size = in_size
        
        self.n1 = 32
        self.n2 = 64
        self.n3 = 128
        
        

        self.ks1 = 3
        self.ks2 = 3
        self.ks3 = 5

        self.c_out = 64
        self.fcl = 32
        
        # Biofluid #1
        self.conv1_1 = nn.Conv1d(3, self.n1, self.ks1 , 2)
        self.batchnorm1_1 = nn.BatchNorm1d(self.n1, False)
        self.maxpool1_1 = nn.MaxPool1d(3)
        
        self.conv2_1 = nn.Conv1d(self.n1, self.n2, self.ks2, 2)
        self.batchnorm2_1 = nn.BatchNorm1d(self.n2, False) # Normalize
        self.maxpool2_1 = nn.MaxPool1d(3)

        self.conv3_1 = nn.Conv1d(self.n2, self.n3, self.ks3, 2)
        self.batchnorm3_1 = nn.BatchNorm1d(self.n3, False)
        self.maxpool3_1 = nn.MaxPool1d(5)
        
        self.fc1 = nn.Linear(128, 256)
        self.activ = nn.Mish()
        self.fc2 = nn.Linear(256, 64)

        self.fusion1 = nn.Linear(64, 32)
        self.fusion2 = nn.Linear(32, 3)

    def forward(self, x):
        # [batch_size, channels, sequence_length]
        # Channels mean the raw signal, first derivative and second derivative
        bsize = x.shape[0]
        x_1 = x.view(bsize, 3, self.in_size)

        # Feature Extraction Module
        x_1 = self.activ(self.conv1_1(x_1))
        x_1 = self.batchnorm1_1(x_1)
        x_1 = self.maxpool1_1(x_1)
        
        x_1 = F.dropout(x_1, 0.5)
        
        x_1 = self.activ(self.conv2_1(x_1))
        x_1 = self.batchnorm2_1(x_1)
        x_1 = self.maxpool2_1(x_1)
        
        x_1 = F.dropout(x_1, 0.5)
        
        x_1 = self.activ(self.conv3_1(x_1))
        x_1 = self.batchnorm3_1(x_1)
        x_1 = self.maxpool3_1(x_1)
        
        
        x_1 = x_1.view(bsize, -1)
        x_1 = F.dropout(x_1, 0.5)

        x_1 = self.activ(self.fc1(x_1))
        x_1 = self.activ(self.fc2(x_1))
        
        x_1 = F.dropout(x_1, 0.25)

        # Fusion Module
        x_1 = self.activ(self.fusion1(x_1))
        x = self.fusion2(x_1)
        
        

        return x


# In[33]:


import os

if not os.path.exists('Ckpts'):
    os.makedirs('Ckpts')


# In[34]:


# Train/Test loop
global_test_acc = []
g_train_score = []
g_test_score = []
g_split = []
run_train_score = []
run_test_score = []
best_binary_pred = []
feats_importance = []
feats_importance_bio_der = []
ite = 0

for lp, ln in zip(labels_pos, labels_neg):
    ite += 1
    bio_pos = extract_X_single_staging(X, 1, bio, patients_matches, quick_labels, lp, info)
    bio_neg = extract_X_single_staging(X, 0, bio, patients_matches, quick_labels, ln, info)
    
    
    X_train = np.concatenate((bio_pos[0], bio_neg[0]))
    y_train = np.concatenate((bio_pos[1], bio_neg[1]))
    X_test = np.concatenate((bio_pos[2], bio_neg[2]))
    y_test = np.concatenate((bio_pos[3], bio_neg[3]))


    
    
    # Uncomment to apply the standard scaler
    # scaler1 = StandardScaler()
    # # transform snv
    # scaler1.fit(X_train[:, 0, :])
    # X_train[:, 0, :] = scaler1.transform(X_train[:, 0, :])
    # X_test[:, 0, :] = scaler1.transform(X_test[:, 0, :])

    # scaler2 = StandardScaler()
    # # transform snv9
    # scaler2.fit(X_train[:, 1, :])
    # X_train[:, 1, :] = scaler2.transform(X_train[:, 1, :])
    # X_test[:, 1, :] = scaler2.transform(X_test[:, 1, :])

    # scaler3 = StandardScaler()
    # # transform snv12
    # scaler3.fit(X_train[:, 2, :])
    # X_train[:, 2, :] = scaler3.transform(X_train[:, 2, :])
    # X_test[:, 2, :] = scaler3.transform(X_test[:, 2, :])
    

    device = 'cuda' 
    tensor_x = torch.Tensor(X_train).to(device) # transform to torch tensor
    tensor_y = torch.Tensor(y_train).to(device)
    
    my_dataset = TensorDataset(tensor_x, tensor_y) # create the dataset
    trainloader = DataLoader(my_dataset, batch_size=64*8, shuffle=True) # create thr dataloader
    
    tensor_x = torch.Tensor(X_test).to(device) # transform to torch tensor
    tensor_y = torch.Tensor(y_test).to(device)
    
    my_dataset = TensorDataset(tensor_x, tensor_y) # create the dataset
    testloader = DataLoader(my_dataset, batch_size=64, shuffle=False) # create the dataloader
    
    
    net = DCNN_one_staging(tensor_x.shape[-1]).to(device) # NN architecture used
    
    
    optimizer = optim.AdamW(net.parameters()) # Optimizer
    
    
    
    # Weights to overcome unbalanced problems
    weights = [1/np.sum(y_train[:, 0]), 1/np.sum(y_train[:, 1]), 1/np.sum(y_train[:, 2])] #[ 1 / number of instances for each class]
    
    class_weights = torch.FloatTensor(weights).cuda()
    
    

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([len(bio_neg[1])/len(bio_pos[1])]).to(device))
    # scaler = torch.cuda.amp.GradScaler()
    best_test_score = 0.0
    best_train_score = 0.0
    iteration = 0
    best_binary_p = np.array([[0], [1]])
    
    
    best_test_score = 0
    best_train_score = 0
    no_improvement_count = 0
    patience = 2000  # Set a patience value
    
    min_diff = float('inf')
    patience_counter = 0
    patience_limit = 500  # You can set this to any desired value
    tolerance = 0.05
    
    
    for epoch in range(5000):  # loop over the dataset multiple times
        # print("\nEpoch ", epoch)
        
        
        net.train()
        
        train_loss = 0.0
        test_loss = 0.0
        test_score = []
        train_score = []
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs)# + torch.rand([inputs.shape[0], inputs.shape[1], 1], device=device)*1 - 0.5)
            
            #Replaces pow(2.0) with abs() for L1 regularization

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            # print statistics
            train_loss += loss.item()
            # binary_prediction = np.round(torch.sigmoid(outputs).cpu().detach().numpy()).astype(int)
            binary_prediction = np.round(torch.softmax(outputs, dim=1).cpu().detach().numpy()).astype(int)
            train_sc = accuracy_score(labels.cpu().detach().numpy(), binary_prediction)
            train_score.append(train_sc)
        
        net.eval()
        
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            outputs = net(inputs)
            # loss = criterion(torch.reshape(outputs, (-1,)), labels) #+ 0.001*reg_loss
            loss = criterion(outputs, labels)

            # print statistics
            test_loss += loss.item()
            
            # binary_prediction = np.round(torch.sigmoid(outputs).cpu().detach().numpy()).astype(int)
            binary_prediction = np.round(torch.softmax(outputs, dim=1).cpu().detach().numpy()).astype(int)
            test_sc = accuracy_score(labels.cpu().detach().numpy(), binary_prediction)
            test_score.append(test_sc)
            
        if np.mean(test_score) >= best_test_score and np.mean(train_score) > best_train_score*0.9999 and epoch > 20:
            best_test_score = np.mean(test_score)
            best_train_score = np.mean(train_score)
            iteration = epoch
            best_binary_p = binary_prediction
            torch.save(net.state_dict(), 'Ckpts/Best.pth')
            print(f' Iter: {ite}, New Best test score found: {best_test_score:.4f} on it: {iteration} with train score: {best_train_score:.4f}')
            # print(f'Train Loss: {train_loss/len(trainloader):.2E}, Test Loss: {test_loss/len(testloader):.2E}')
            no_improvement_count = 0  # Reset the no_improvement_count since the test score has improved
        else:
            no_improvement_count += 1  # Increment the no_improvement_count since the test score has not improved
        
        
#         diff = np.mean(train_score) - np.mean(test_score)

#         if diff < (min_diff - tolerance):
#             min_diff = diff
#             patience_counter = 0
#             # Save your best model here
#         else:
#             patience_counter += 1
            
#         # If patience counter reaches the limit, stop training
#         if patience_counter >= patience_limit:
#             print(f'Patience EARLY STOP: Best test score: {best_test_score:.4f} on it: {iteration} with train score: {np.mean(train_score):.4f}')
#             break
            
        if best_test_score>0.95 and np.mean(train_score)>0.9:
            torch.save(net.state_dict(), 'Ckpts/Best.pth')
            print(f'EARLY STOP: Best test score: {best_test_score:.4f} on it: {iteration} with train score: {np.mean(train_score):.4f}')
            
            break
#         if no_improvement_count >= patience:
#             print(f'Early stopping after {patience} epochs with no improvement')
#             break
        

        run_train_score.append(np.mean(train_score))
        run_test_score.append(np.mean(test_score))

        # print(f'Train Loss: {train_loss/len(trainloader)}, Test Loss: {test_loss/len(testloader)}')
        # print(f'Mean metric train: {np.mean(train_score)}')
        # print(f'Mean metrics: {np.mean(train_score)}, {np.mean(test_score)}')
        
    # best_binary_pred.append([best_binary_p[:,0], y_test])
    print(lp)
    print(ln)
    print(best_binary_p)

    best_binary_pred.append([best_binary_p, y_test])

    g_split.append([len(y_test), np.sum(y_test)])
    g_train_score.append(best_train_score)
    g_test_score.append(best_test_score)
    print(f' Mean metric train: {np.mean(g_train_score):.4f}, Mean metric test: {np.mean(g_test_score):.4f}, Best metric test: {best_test_score:.4f}, {iteration}')
    #%%
    net.load_state_dict(torch.load('Ckpts/Best.pth'))
    feat_imp, fi_bio_der = obtain_features_importance(net, X_test, y_test)

    feats_importance.append(feat_imp)
    feats_importance_bio_der.append(fi_bio_der)
    
np.savetxt(f"Results/feature_importance_single_{bio}.csv", feats_importance, delimiter=",")
feats_importance_bio_der = np.array(feats_importance_bio_der)
np.savetxt(f"Results/feature_importance_single_{bio}_derv.csv", feats_importance_bio_der, delimiter=",")


# In[20]:


best_model = DCNN_one_staging(tensor_x.shape[-1]).to(device)
best_model.load_state_dict(torch.load('Ckpts/Best.pth'))
best_model.eval()


# In[21]:


num_samples = 100
num_channels = 3
num_data_points = 830

random_data = np.random.rand(num_samples, num_channels, num_data_points)

num_classes = 3
random_labels = np.random.randint(0, 2, size=(num_samples, num_classes))
tensor_x_random = torch.Tensor(random_data).to(device)
tensor_y_random = torch.Tensor(random_labels).to(device)

random_dataset = TensorDataset(tensor_x_random, tensor_y_random)
random_dataloader = DataLoader(random_dataset, batch_size=64, shuffle=False)

random_test_score = []

for i, data in enumerate(random_dataloader, 0):
    inputs, labels = data
    outputs = best_model(inputs)
    binary_prediction = np.round(torch.softmax(outputs, dim=1).cpu().detach().numpy()).astype(int)
    test_sc = accuracy_score(labels.cpu().detach().numpy(), binary_prediction)
    random_test_score.append(test_sc)

print(f'Random data test score: {np.mean(random_test_score):.4f}')


# In[22]:


train_test_score = []

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    outputs = best_model(inputs)
    binary_prediction = np.round(torch.softmax(outputs, dim=1).cpu().detach().numpy()).astype(int)
    test_sc = accuracy_score(labels.cpu().detach().numpy(), binary_prediction)
    train_test_score.append(test_sc)

print(f'Train data test score: {np.mean(train_test_score):.4f}')


# In[23]:


if not os.path.exists('Results'):
    os.makedirs('Results')
np.savetxt(f"Results/feature_importance_single_{bio}.csv", feats_importance, delimiter=",")
feats_importance_bio_der = np.array(feats_importance_bio_der)
np.savetxt(f"Results/feature_importance_single_{bio}_derv.csv", feats_importance_bio_der, delimiter=",")


# In[24]:


# Confusion matrix
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# y_preds  = np.array(best_binary_pred)[:, 0]
# y_target = np.array(best_binary_pred)[:, 1]

# y_preds2 = np.array([item for sublist in y_preds for item in sublist])
# y_target2 = np.array([item for sublist in y_target for item in sublist])

# print(confusion_matrix(y_target2.argmax(axis=1), y_preds2.argmax(axis=1)))


# In[25]:


# from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
y_preds = []
y_target = []
for i in range(len(best_binary_pred)):
    y_preds.append(best_binary_pred[i][0])
    y_target.append(best_binary_pred[i][1])

y_preds2 = np.concatenate(y_preds)
y_target2 = np.concatenate(y_target)


# In[26]:


print(confusion_matrix(y_target2.argmax(axis=1), y_preds2.argmax(axis=1)))


# In[ ]:





# In[ ]:





# In[ ]:




