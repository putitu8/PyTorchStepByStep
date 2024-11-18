from config import *
config_chapter3()
# This is needed to render the plots in this chapter
from plots.chapter3 import *

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

from stepbystep.v0 import StepByStep

X, y = make_moons(n_samples=100, noise=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=13)

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_val = sc.transform(X_val)

fig = figure1(X_train, y_train, X_val, y_val)

#Data Preparation

torch.manual_seed(13)

# Builds tensors from numpy arrays
x_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()

x_val_tensor = torch.as_tensor(X_val).float()
y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()

# Builds dataset containing ALL data points
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# Builds a loader of each set
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16)

def odds_ratio(prob):
    return prob / (1 - prob)

def log_odds_ratio(prob):
    return np.log(odds_ratio(prob))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Sets learning rate - this is "eta" ~ the "n" like Greek letter
lr = 0.1

torch.manual_seed(42)
model = nn.Sequential()
model.add_module('linear', nn.Linear(2, 1))

# Defines a SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines a BCE loss function
loss_fn = nn.BCEWithLogitsLoss()

n_epochs = 100

sbs = StepByStep(model, loss_fn, optimizer)
sbs.set_loaders(train_loader, val_loader)
sbs.train(n_epochs)

fig = sbs.plot_losses()
plt.figure(fig)
plt.show()

print(model.state_dict())


logits_val = sbs.predict(X_val)
print(logits_val.squeeze())
probabilities_val = sigmoid(logits_val).squeeze()
print(probabilities_val)


#标准阀值，50%
cm_thresh50 = confusion_matrix(y_val, (probabilities_val >= .5))
print(cm_thresh50)

#降低了阈值，所有样本都被预测为正类
cm_thresh0 = confusion_matrix(y_val, (probabilities_val >= 0))
print(cm_thresh0)

#提高了阈值，所有样本都被预测为负类
cm_thresh100 = confusion_matrix(y_val, (probabilities_val >= 1))
print(cm_thresh100)

def split_cm(cm):
    # Actual negatives go in the top row, 
    # above the probability line
    actual_negative = cm[0]
    # Predicted negatives go in the first column
    tn = actual_negative[0]
    # Predicted positives go in the second column
    fp = actual_negative[1]

    # Actual positives go in the bottow row, 
    # below the probability line
    actual_positive = cm[1]
    # Predicted negatives go in the first column
    fn = actual_positive[0]
    # Predicted positives go in the second column
    tp = actual_positive[1]
    
    return tn, fp, fn, tp

def tpr_fpr(cm):
    tn, fp, fn, tp = split_cm(cm)
    
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    return tpr, fpr

def precision_recall(cm):
    tn, fp, fn, tp = split_cm(cm)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return precision, recall


threshs = np.linspace(0.,1,11)
fig = figure17(y_val, probabilities_val, threshs)
plt.figure(fig)
plt.show()

fpr, tpr, thresholds1 = roc_curve(y_val, probabilities_val)
prec, rec, thresholds2 = precision_recall_curve(y_val, probabilities_val)
fig = eval_curves(fpr, tpr, rec, prec, thresholds1, thresholds2, line=True)
plt.figure(fig)
plt.show()

auroc = auc(fpr, tpr)
aupr = auc(rec, prec)
print(auroc, aupr)