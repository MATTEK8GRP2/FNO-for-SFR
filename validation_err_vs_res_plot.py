# -*- coding: utf-8 -*-
"""


Created on Tue Apr 27 16:43:38 2021


Authors:  Martin Voigt Vejling, Simone Birk Bols Thomsen and
          Kristian Søgaard
E-Mails: {mvejli17, sbbt17, ksagaa17}@student.aau.dk

In this script the FNO and the baseline model are evaluated on the different
validation data sets as described in the report

        Fourier Neural Operators for Sound Field Reconstruction
                        in the Time Domain
           - Chapter 7: Temporal Sound Field Reconstruction
                    
The models used in this script can be made using the
script train_FNO_network_2d.py

The script has been developed using Python 3.6 with the libraries numpy, scipy,
and torch 1.7.0. For plotting purposes the matplotlib.pyplot package has been
used.


"""

from scipy.io import loadmat
import numpy as np
import torch
import matplotlib.pyplot as plt

from data.input_data_generation import create_input_tensor
from Loss_Function import LpLoss
from Normalization import Max_Normalization


def eval_u(VALIDATION_PATH, model_name, targets_between_input, normalizer, a_downsample,
           no_cuda=False):
    # =============================================================================
    # Load data
    # =============================================================================
    batch_size2 = 20

    device = torch.device('cpu')
    normalizer.to(device)
    kwargs = {"targets_between_input": targets_between_input,
              "Nt": 1201,
              "fs": 1200,
              "a_downsample": a_downsample}

    dict_ = loadmat(VALIDATION_PATH+'validation_a.mat')
    A_p = dict_["a"]
    nval = np.shape(A_p)[0]
    A_p = A_p.astype(np.float32)
    A_p = torch.from_numpy(A_p)
    A_p_norm = normalizer.encode(A_p)

    dict_ = loadmat(VALIDATION_PATH+'validation_x.mat')
    x = dict_["x"]
    x = x.astype(np.float32)
    x = torch.from_numpy(x)

    U = loadmat(VALIDATION_PATH+'validation_u{}.mat'.format(targets_between_input))["u"]
    U = U.astype(np.float32)
    U = torch.from_numpy(U)
    U_norm = normalizer.encode(U)

    A_norm = create_input_tensor(A_p_norm, x, **kwargs)
    validation_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(A_norm, U_norm),
                                                    batch_size=batch_size2, shuffle=False)


    # =============================================================================
    # Testing
    # =============================================================================

    model = torch.load('model/'+model_name)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    myloss = LpLoss(size_average=False, reduction=True)
    normalizer.to(device)
    validation_l2 = 0
    model.eval()
    with torch.no_grad():
        for A_batch, U_batch in validation_loader:
            A_batch, U_batch = A_batch.to(device), U_batch.to(device)

            U_batch_hat = model(A_batch)

            U_batch = normalizer.decode(U_batch)
            U_batch_hat = normalizer.decode(U_batch_hat)

            validation_l2 += myloss(U_batch_hat.view(batch_size2, -1), U_batch.view(batch_size2, -1)).item()

    validation_l2 /= nval
    return validation_l2


if __name__ == '__main__':
    ### Construct Normalizer ###
    norm_data = loadmat("data/max_normalization{}.mat".format(1000))
    p_max = norm_data["p_max"]
    normalizer = Max_Normalization(p_max)


    # =============================================================================
    # Specifications
    # =============================================================================

    no_cuda = False
    targets_between_input_list = [1, 2, 3, 5, 11, 23]
    a_downsample = 24
    model_name = 'FNO001'
    #model_name = 'Baseline001'
    VALIDATION_PATH = 'data/validation_data/'


    # =============================================================================
    # Evaluate Losses on the Validation Data Sets
    # =============================================================================

    validation_l2_list = []
    for targets_between_input in targets_between_input_list:
        validation_l2 = eval_u(VALIDATION_PATH, model_name, targets_between_input,
                               normalizer, a_downsample, no_cuda)
        validation_l2_list.append(validation_l2)


    # =============================================================================
    # Plot
    # =============================================================================

    reconstructed_grid_time = [25*i for i in targets_between_input_list]

    plt.plot(reconstructed_grid_time, validation_l2_list, marker='o')
    plt.xlabel('Time Resolution')
    plt.ylabel('Relative $\ell_2$ Error')
    plt.xticks(reconstructed_grid_time, rotation=90)
    plt.ylim(bottom=3e-02, top=2*1e-00)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig('results/Train/'+model_name+'_err_vs_res.png')
    plt.show()

    with open('results/Train/'+model_name+'_rel_err_list.txt', 'w') as f:
        f.write('u1\t u2\t u3\t u5\t u11\t u23\n')
        for l2 in validation_l2_list:
            f.write('{}'.format(str(l2)))
            f.write('   &   ')
    print(validation_l2_list)
