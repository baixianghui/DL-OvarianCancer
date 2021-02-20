# -*- coding: utf-8 -*-


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import roc_auc_score,roc_curve
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle
import scipy.stats
from scipy import stats
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


class OvarianDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ovarian_frame = pd.read_csv(csv_file)
        
        self.transform = transform
        self.classes = ['neg','pos']
        self.imgs = []
        
        self.labels = self.ovarian_frame.ix[:,5].values
        
        for idx in range(len(self.ovarian_frame)):
            self.imgs.append((self.ovarian_frame.iloc[idx,0],self.ovarian_frame.iloc[idx,5]))
            

    def __len__(self):
        return len(self.ovarian_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        image1 = Image.open(self.ovarian_frame.iloc[idx,0])
        image2 = Image.open(self.ovarian_frame.iloc[idx,1])
        image3 = Image.open(self.ovarian_frame.iloc[idx,2])
        image4 = Image.open(self.ovarian_frame.iloc[idx,3])
        image5 = Image.open(self.ovarian_frame.iloc[idx,4])
        label = np.array(self.labels[idx],dtype=np.int64)
        
        
        
        sample = {'image': [image1,image2,image3,image4,image5], 'label': label}
        if self.transform:
            sample['image'][0] = self.transform(sample['image'][0])
            sample['image'][1] = self.transform(sample['image'][1])
            sample['image'][2] = self.transform(sample['image'][2])
            sample['image'][3] = self.transform(sample['image'][3])
            sample['image'][4] = self.transform(sample['image'][4])

        return sample



def inference(model, device, img_name1, img_name2,img_name3, img_name4,img_name5, transform):
    model.eval()
    image1 = transform(Image.open(img_name1))
    image2 = transform(Image.open(img_name2))
    image3 = transform(Image.open(img_name3))
    image4 = transform(Image.open(img_name4))
    image5 = transform(Image.open(img_name5))
    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)
    image3 = image3.unsqueeze(0)
    image4 = image4.unsqueeze(0)
    image5 = image5.unsqueeze(0)
    image1 = image1.to(device)
    image2 = image2.to(device)
    image3 = image3.to(device)
    image4 = image4.to(device)
    image5 = image5.to(device)
    with torch.set_grad_enabled(False):
    	output = (model(image1,image2,image3,image4,image5))
    	pred = nn.Softmax(dim=1)(output).cpu().numpy()[0][1]
    
    return pred

def train_model(model, criterion, dataloaders, device, dataset_sizes, optimizer, scheduler, num_epochs=50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0

    
    rec_train_loss = []
    rec_val_loss = []
    rec_train_acc = []
    rec_val_acc = []
    rec_epoch = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        rec_epoch.append(epoch)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_preds = []
            running_labels = []

            # Iterate over data.
            for sample in dataloaders[phase]:
                images0 = sample['image'][0]
                images1 = sample['image'][1]
                images2 = sample['image'][2]
                images3 = sample['image'][3]
                images4 = sample['image'][4]
                labels = sample['label']
                if phase == 'Val':
                    running_labels.append(labels.numpy()[0])
                #extra_fts = extra_fts.float().to(device)
                images0 = images0.to(device)
                images1 = images1.to(device)
                images2 = images2.to(device)
                images3 = images3.to(device)
                images4 = images4.to(device)
                labels = labels.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = (model(images0,images1,images2,images3,images4))
                    #labels = torch.squeeze(labels)

                    _, preds = torch.max(outputs, 1)
                    #preds = (outputs > 0.5).int()
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                    if phase == 'Val':
                        score = nn.Softmax(dim=1)(outputs)
                        running_preds.append(score.cpu().numpy()[0][1])

                # statistics
                running_loss += loss.item() * images0.size(0)

                running_corrects += torch.sum(preds == labels.data)

            if phase == 'Train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            
            if phase == 'Train':
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                rec_train_loss.append(epoch_loss)
                rec_train_acc.append(epoch_acc)
                
            else:
                auc = roc_auc_score(np.array(running_labels),np.array(running_preds))
                print ('{} Loss: {:.4f} auc: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,auc,epoch_acc))
                rec_val_loss.append(epoch_loss)
                rec_val_acc.append(epoch_acc)
                
            # deep copy the model
            if phase == 'Val' and auc > best_auc:#epoch_acc > best_acc:
                best_auc = auc#+epoch_acc#epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        


    time_elapsed = time.time() - since
    
    
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()
        
        self.cnn1 = models.resnet18(pretrained=True)
        
        self.cnn1.fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(self.cnn1.fc.in_features, 128))
        self.cnn2 = models.resnet18(pretrained=True)
        
        self.cnn2.fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(self.cnn2.fc.in_features, 256))
        self.cnn3 = models.resnet18(pretrained=True)
        
        self.cnn3.fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(self.cnn3.fc.in_features, 128))
        
        
        self.fc3 = nn.Linear(768, 32)
        self.fc4 = nn.Linear(32, 2)
        
        
    def forward(self, image1, image2, image3,image4,image5):
        x1 = self.cnn1(image1)
        x2 = self.cnn1(image2)
        x3 = self.cnn2(image3)
        x4 = self.cnn3(image4)
        x5 = self.cnn3(image5)
        
        x = torch.cat((x1, x2), dim=1)
        x = torch.cat((x, x3), dim=1)
        x = torch.cat((x, x4), dim=1)
        x = torch.cat((x, x5), dim=1)
        
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc4(x)
        return x





def run(seed):
    setup_seed(seed)
    data_transforms = {
        'Train': transforms.Compose([
            #transforms.RandomResizedCrop(224),
            #transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
            #transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ]),
        'Val': transforms.Compose([
            #transforms.Resize((224,224)),
            #transforms.CenterCrop(299),
            transforms.ToTensor()
            #transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ]),
        'Test': transforms.Compose([
            #transforms.Resize((224,224)),
            #transforms.CenterCrop(299),
            transforms.ToTensor()
            #transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ]),
    }


    image_datasets = {x: OvarianDataset(x+'.csv',transform=data_transforms[x])
                      for x in ['Train', 'Val','Test']}

    num_classes = len(image_datasets['Train'].classes)
    class_sample_counts = [0] * num_classes                                                      

    for item in image_datasets['Train'].imgs:
        class_sample_counts[item[1]] += 1

    class_weights = 1./torch.Tensor(class_sample_counts)
    train_targets = [sample[1] for sample in image_datasets['Train'].imgs]
    train_samples_weight = [class_weights[class_id] for class_id in train_targets]
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, len(image_datasets['Train']))
    dataloaders = dict()
    dataloaders['Train'] = torch.utils.data.DataLoader(image_datasets['Train'], batch_size=16, shuffle = False,                              
                                                                 sampler = train_sampler, num_workers=4, pin_memory=True,drop_last=True)  
    dataloaders['Val'] = torch.utils.data.DataLoader(image_datasets['Val'], batch_size=1,
                                                 shuffle=False, num_workers=4)
    dataloaders['Test'] = torch.utils.data.DataLoader(image_datasets['Test'], batch_size=1,
                                                 shuffle=False, num_workers=4)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Val','Test']}
    class_names = image_datasets['Train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = HybridNet()
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()#nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9,weight_decay=0.001)
    #optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.0001,weight_decay=0)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    model_ft = train_model(model_ft, criterion, dataloaders, device, dataset_sizes,optimizer_ft, exp_lr_scheduler,num_epochs=10)

    torch.save(model_ft.state_dict(), 'ovarian.pth')
    model_ft.load_state_dict(torch.load('ovarian.pth'))
    annotations = pd.read_csv('Test.csv')
    running_corrects = 0
    scores = []
    labels = []
    pairs = dict()
    benign_corrects = 0
    benign_labels = 0
    malig_corrects = 0
    malig_labels = 0
    thresh = 0.6
    pred_result = dict()
    gts = dict()

    for i in range(len(annotations)):
        img_name1 = annotations.iloc[i,0]
        img_name2 = annotations.iloc[i,1]
        img_name3 = annotations.iloc[i,2]
        img_name4 = annotations.iloc[i,3]
        img_name5 = annotations.iloc[i,4]
        label = annotations.iloc[i,5]
        #print (img_name)
        #print (label)
        pred = inference(model_ft,device,img_name1,img_name2,img_name3,img_name4,img_name5,data_transforms['Test'])

        if img_name1 not in pred_result:
            pred_result[img_name1] = []

        if img_name1 not in gts:
            gts[img_name1] = label

        #scores.append(pred)
        pred_result[img_name1].append(pred)
     

    for key in pred_result:

        scores.append(np.max(pred_result[key]))
        labels.append(gts[key])
        pairs[key] = [gts[key],np.max(pred_result[key])]
        running_corrects += int((np.max(pred_result[key])>thresh)==gts[key])
        if gts[key] == 0:
            benign_labels += 1
            if np.max(pred_result[key]) <= thresh:
                benign_corrects += 1
        else:
            malig_labels += 1
            if np.max(pred_result[key]) >thresh:
                malig_corrects +=1

    alpha = .95
    auc, auc_cov = delong_roc_variance(
        np.array(labels),
        np.array(scores))

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1

    print('AUC:', auc)
    print('AUC COV:', auc_cov)
    print('95% AUC CI:', ci)
    fpr, tpr, thresholds = roc_curve(np.array(labels), np.array(scores))
    
    plt.plot(fpr,tpr)
    plt.show()

    with open('pred_result.txt','w') as f:
        for key in pairs:
            key1=key.split('/')[-1].split('.')[0]
            f.write(key1+',' + str(pairs[key][0]) + ',' + str(pairs[key][1])+'\n')
    
    
    print ('CorrectPreds: {} TotalPreds: {} auc: {:.4f} Acc: {:.4f}'.format(running_corrects,len(labels),auc,float(running_corrects)/float(len(labels))))
    print ('CorrectBenignPreds: {} TotalBenignPreds: {} specificity: {:.4f}'.format(benign_corrects,benign_labels,float(benign_corrects)/float(benign_labels)))
    print ('CorrectMalignantPreds: {} TotalMalignantPreds: {} sensitivity: {:.4f}'.format(malig_corrects,malig_labels,float(malig_corrects)/float(malig_labels)))

    return auc

if __name__ == "__main__":
    seed = 44
    run(seed)


