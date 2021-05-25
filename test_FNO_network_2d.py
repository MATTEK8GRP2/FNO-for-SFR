# -*- coding: utf-8 -*-
"""


Created on Mon Apr 19 15:17:19 2021


Authors:  Martin Voigt Vejling, Simone Birk Bols Thomsen and
          Kristian Søgaard
E-Mails: {mvejli17, sbbt17, ksagaa17}@student.aau.dk

In this script the FNO and the baseline model are evaluated on the different
test data sets as described in the report

        Fourier Neural Operators for Sound Field Reconstruction
                        in the Time Domain
               - Chapter 8: Numerical Experiments

The models used in this script can be made with the
script train_FNO_network_2d.py

The script has been developed using Python 3.6 with the libraries numpy, scipy,
and torch 1.7.0. For plotting purposes the matplotlib.pyplot package has been
used.


"""


import numpy as np
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt

from data.input_data_generation import create_input_tensor
from Loss_Function import LpLoss
from Normalization import Max_Normalization


def test_u(TEST_PATH, model_name, targets_between_input, normalizer, a_downsample,
           no_cuda=False):
    # =============================================================================
    # Load data
    # =============================================================================
    device = torch.device('cpu')
    normalizer.to(device)
    kwargs = {"targets_between_input": targets_between_input,
              "Nt": 1201,
              "fs": 1200,
              "a_downsample": a_downsample}

    dict_ = loadmat(TEST_PATH+'test_a.mat')
    A_p = dict_["a"]
    ntest = np.shape(A_p)[0]
    A_p = A_p.astype(np.float32)
    A_p = torch.from_numpy(A_p)
    A_p_norm = normalizer.encode(A_p)

    dict_ = loadmat(TEST_PATH+'test_x.mat')
    x = dict_["x"]
    x = x.astype(np.float32)
    x = torch.from_numpy(x)

    U = loadmat(TEST_PATH+'test_u{}.mat'.format(targets_between_input))["u"]
    U = U.astype(np.float32)
    U = torch.from_numpy(U)
    U_norm = normalizer.encode(U)

    A_norm = create_input_tensor(A_p_norm, x, **kwargs)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(A_norm, U_norm),
                                              batch_size=20, shuffle=False)

    # =============================================================================
    # Testing
    # =============================================================================
    model = torch.load('model/'+model_name)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    myloss = LpLoss(size_average=False, reduction=True)
    normalizer.to(device)

    test_l2 = 0
    model.eval()
    with torch.no_grad():
        for A_batch, U_batch in test_loader:
            A_batch, U_batch = A_batch.to(device), U_batch.to(device)

            U_batch_hat = model(A_batch)
            U_batch = normalizer.decode(U_batch)
            U_batch_hat = normalizer.decode(U_batch_hat)

            test_l2 += myloss(U_batch_hat.view(20, -1), U_batch.view(20, -1)).item()

    test_l2 /= ntest
    return test_l2


if __name__ == '__main__':
    ### Construct Normalizer ###
    norm_data = loadmat("data/max_normalization{}.mat".format(1000))
    p_max = norm_data["p_max"]
    normalizer = Max_Normalization(p_max)


    # =============================================================================
    # Specifications
    # =============================================================================

    no_cuda = False
    a_downsample = 24
    targets_between_input_list = [1, 2, 3, 5, 11, 23]
    model_name = 'FNO001'
    model_baseline = 'Baseline001'
    TEST_PATH = 'data/test_data/'


    # =============================================================================
    # Evaluate Loss on the Test Data Sets
    # =============================================================================

    test_l2_list = []
    for targets_between_input in targets_between_input_list:
        test_l2 = test_u(TEST_PATH, model_name, targets_between_input, normalizer, a_downsample, no_cuda)
        test_l2_list.append(test_l2)

    if model_baseline != None:
        baseline_test_l2_list = []
        for targets_between_input in targets_between_input_list:
            baseline_test_l2 = test_u(TEST_PATH, model_baseline, targets_between_input, normalizer, a_downsample, no_cuda)
            baseline_test_l2_list.append(baseline_test_l2)


    # =============================================================================
    # Plot
    # =============================================================================

    reconstructed_grid_time = [25*i for i in targets_between_input_list]

    plt.plot(reconstructed_grid_time, test_l2_list, marker='o', label='FNO')
    plt.plot(reconstructed_grid_time, baseline_test_l2_list, marker='o', label='Baseline')
    if model_baseline != None:
        plt.legend(loc='lower right')
    plt.xlabel('Time Resolution')
    plt.ylabel('Relative $\ell_2$ Error')
    plt.xticks(reconstructed_grid_time, rotation=90)
    plt.ylim(bottom=3e-02, top=2*1e-00)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig('results/Test/'+model_name+'_err_vs_res_w_baseline.png', bbox_inches = 'tight')
    plt.show()

    with open('results/Test/'+model_name+'_rel_err_list.txt', 'w') as f:
        f.write('u1\t u2\t u3\t u5\t u11\t u23\n')
        for l2 in test_l2_list:
            f.write('{}'.format(str(l2)))
            f.write('   &   ')

    if model_baseline != None:
        with open('results/Test/'+model_baseline+'_rel_err_list.txt', 'w') as f:
            f.write('u1\t u2\t u3\t u5\t u11\t u23\n')
            for l2 in baseline_test_l2_list:
                f.write('{}'.format(str(l2)))
                f.write('   &   ')

print(test_l2_list)
print(baseline_test_l2_list)
