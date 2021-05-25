# -*- coding: utf-8 -*-
"""


Created on Mon Apr 19 13:22:32 2021


Authors:  Martin Voigt Vejling, Simone Birk Bols Thomsen and
          Kristian Søgaard
E-Mails: {mvejli17, sbbt17, ksagaa17}@student.aau.dk

In this script the FNO and the baseline model can be trained using
the training data while also computing the validation loss during training.
The training is in line with the description in the report

        Fourier Neural Operators for Sound Field Reconstruction
                        in the Time Domain
           - Chapter 7: Temporal Sound Field Reconstruction
                    
The models made with this script can be used to evaluate on the validation
and the test data sets in the scripts validation_err_vs_res_plot.py and
test_FNO_network_2d.py.

The models being trained is found in the scripts FNO_network_2d.py and
Baseline_Network.py.

The script has been developed using Python 3.6 with the libraries numpy, scipy,
and torch 1.7.0. For plotting purposes the matplotlib.pyplot package has been
used.


"""

import json
import os.path

import torch
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from data.input_data_generation import create_input_tensor
import networks.FNO_network_2d as MyNetwork
import networks.Baseline_Network as Baseline
from Normalization import Max_Normalization
from Loss_Function import LpLoss


################################################################
# Configurations
################################################################

seed = 10 # random seed
torch.manual_seed(seed)
np.random.seed(seed)
no_cuda = False

TRAIN_PATH = 'data/train_data/'
VALID_PATH = 'data/validation_data/'

ntrain = 1000
nval = 400

log_status = 1

model = "FNO"
#model = "Baseline"

modes_x = 6 # maximum 8 due to the use of only 16 microphones
modes_t = 12 # maximum N_u/2 (75/2 for training)
width = 64 # greater than 28
q_width = 128

batch_size = 10
batch_size2 = 1

epochs = 1
learning_rate = 0.001
scheduler_step = 30
scheduler_gamma = 0.5
weight_decay = 1e-3

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

path = model+'002' # Name of the model used for saving model and results
path_model = 'model/'+path
path_train_err = 'results/Train/'+path+'_train.png'
path_train_log = 'results/Train/'+path+'_log.txt'

assert not os.path.isfile(path_train_log) # Don't overwrite log files


################################################################
# Load Data
################################################################

norm_data = loadmat("data/max_normalization{}.mat".format(ntrain))
p_max = norm_data["p_max"]
normalizer = Max_Normalization(p_max)

kwargs = {"targets_between_input": 3, "Nt": 1201, "fs": 1200, "a_downsample": 24}

### Training data ###
dict_ = loadmat(TRAIN_PATH+'train_a.mat')
A_p = dict_["a"]
A_p = A_p.astype(np.float32)
A_p = torch.from_numpy(A_p)
A_p_norm = normalizer.encode(A_p)

dict_ = loadmat(TRAIN_PATH+'train_x.mat')
x = dict_["x"]
x = x.astype(np.float32)
x = torch.from_numpy(x)

U = loadmat(TRAIN_PATH+'train_u3.mat')["u"]
U = U.astype(np.float32)
U = torch.from_numpy(U)
U_norm = normalizer.encode(U)

A_norm = create_input_tensor(A_p_norm, x, **kwargs)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(A_norm, U_norm),
                                           batch_size=batch_size, shuffle=True)

### Validation data ###
dict_ = loadmat(VALID_PATH+'validation_a.mat')
A_p = dict_["a"]
A_p = A_p.astype(np.float32)
A_p = torch.from_numpy(A_p)
A_p_norm = normalizer.encode(A_p)

dict_ = loadmat(VALID_PATH+'validation_x.mat')
x = dict_['x']
x = x.astype(np.float32)
x = torch.from_numpy(x)

U = loadmat(VALID_PATH+'validation_u3.mat')["u"]
U = U.astype(np.float32)
U = torch.from_numpy(U)
U_norm = normalizer.encode(U)

A_norm = create_input_tensor(A_p_norm, x, **kwargs)
valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(A_norm, U_norm),
                                           batch_size=batch_size2, shuffle=False)


################################################################
# Prepare Model for Training
################################################################

use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
normalizer.to(device)

if model == "FNO":
    model = MyNetwork.Net2d(modes_x, modes_t, width, q_width).to(device)
elif model == "Baseline":
    model = Baseline.Net2d(modes_x, modes_t, width, q_width).to(device)

model.summary_of_model()
model_params = model.count_params()
print(model_params)

### Save Parameter Settings to Log ###
hyper_parameters = {"modes_x": modes_x,
                    "modes_t": modes_t,
                    "width": width,
                    "q_width": q_width,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "scheduler_step": scheduler_step,
                    "scheduler_gamma": scheduler_gamma,
                    "weight_decay": weight_decay,
                    "random_seed": seed}
with open(path_train_log, 'w+') as f:
    f.write(path+' Log\n\n')
    f.write(json.dumps(hyper_parameters))
    f.write('\n')
    f.write('{}\n'.format(model_params))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False, reduction=True)
myavgloss = LpLoss(size_average=True, reduction=True)


################################################################
# Training and Evaluation
################################################################

train_loss = []
valid_loss = []
valid_loss_u2 = []

for ep in range(epochs):
    model.train()
    train_l2 = 0
    for A_batch, U_batch in train_loader:
        A_batch, U_batch = A_batch.to(device), U_batch.to(device)

        optimizer.zero_grad()
        U_batch_hat = model(A_batch)

        l2_enc = myavgloss(U_batch_hat.view(batch_size, -1), U_batch.view(batch_size, -1))
        l2_enc.backward() # Backpropagation

        U_batch = normalizer.decode(U_batch)
        U_batch_hat = normalizer.decode(U_batch_hat)
        l2 = myloss(U_batch_hat.view(batch_size, -1), U_batch.view(batch_size, -1))

        optimizer.step()

        train_l2 += l2.item()

    scheduler.step()

    if ep % log_status == 0:
        model.eval()
        valid_l2 = 0.0
        valid_l2_u2 = 0.0
        with torch.no_grad():
            for A_batch, U_batch in valid_loader:
                A_batch, U_batch = A_batch.to(device), U_batch.to(device)

                U_batch_hat = model(A_batch)

                U_batch = normalizer.decode(U_batch)
                U_batch_hat = normalizer.decode(U_batch_hat)
                valid_l2 += myloss(U_batch_hat.view(batch_size2, -1), U_batch.view(batch_size2, -1)).item()

        train_l2 /= ntrain
        valid_l2 /= nval

        print('Epoch: {:5s} \t Learning Rate: {:.2e} \tTrain Loss:  {:.3e} \
              Validation Loss:  {:.3e}'.format(str(ep),
              optimizer.param_groups[0]['lr'], train_l2, valid_l2))

        with open(path_train_log, 'a') as f:
            f.write('\nEpoch: {:5s} \t Learning Rate: {:.2e} \tTrain Loss:  {:.3e} \
                    Validation Loss:  {:.3e}'.format(str(ep),
                    optimizer.param_groups[0]['lr'], train_l2, valid_l2))

        train_loss.append(train_l2)
        valid_loss.append(valid_l2)

torch.save(model, path_model)

# =============================================================================
# Plot
# =============================================================================

xaxis = np.arange(0, epochs, log_status)
plt.plot(xaxis, train_loss, '-g', label='Training loss')
plt.plot(xaxis, valid_loss, '-m', label='Validation loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Relative $\ell_2$ Error')
plt.yscale("log")
plt.tight_layout()
plt.savefig(path_train_err, bbox_inches='tight')
plt.show()

