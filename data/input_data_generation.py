# -*- coding: utf-8 -*-
"""


Created on Wed Apr 14 11:29:51 2021


Authors:  Martin Voigt Vejling, Simone Birk Bols Thomsen and
          Kristian Søgaard
E-Mails: {mvejli17, sbbt17, ksagaa17}@student.aau.dk

This script contains functionality to form training, testing, and validation
data sets by importing simulated room impulse responses with the script
RIR_generator. This functionality in included in the function input_data
which is run in this script. Moreover, functionality to form the input
tensor A as described in the report

        Fourier Neural Operators for Sound Field Reconstruction
                        in the Time Domain
           - Chapter 7: Temporal Sound Field Reconstruction
                    
The data sets made with this script is used for training, validating, and
testing the FNO and the baseline model as described in 

                  - Chapter 8: Numerical Experiments

This is done using the scripts train_FNO_network_2d.py,
validation_err_vs_res_plot.py, and test_FNO_network_2d.py.

The script has been developed using Python 3.6 with the
libraries numpy, scipy, and torch 1.7.0.


"""

import numpy as np
from math import ceil
from scipy.io import loadmat, savemat
import torch


def input_data(PATH, n_mics=32, sub=2, Nt=1201, fs=1200,
               a_downsample=24, targets_between_input=3):
    """
    Method used to create training, testing, and validation data sets.

    Parameters
    ----------
    PATH : str
        The path to where the simulated room impulse responses are saved
        in a .mat file.
    n_mics : int, optional
        Number of microphones used in the simulation. The default is 32.
    sub : int, optional
        Downsampling in the spatial axis. The default is 2.
    Nt : int, optional
        Number of temporal sample points used in the simulation.
        The default is 1201.
    fs : int, optional
        The sample frequency used in the simulation. The default is 1200.
    a_downsample : int, optional
        Downsampling in the temporal axis for the available temporal
        sample points in the input. The default is 24.
    targets_between_input : int, optional
        The number of target time locations between the input time locations.
        The default is 3.

    Returns
    -------
    a : ndarray, size=(n_sims, N_x, N_a)
        Impulse response measurements in the input used to form the tensor A.
    u : ndarray, size=(n_sims, N_x, N_u)
        Impulse response measurements in the target forming the matrix U.
    x : ndarray, size=(n_sims, N_x)
        Spatial position of microphones.

    """

    assert targets_between_input in [1, 2, 3, 5, 11, 23] # Viable targets_between_input

    Nt_half = ceil(Nt/2)
    u_downsample = ceil((a_downsample-1)/(targets_between_input + 1))

    data = loadmat(PATH)
    a = data['h'][:, :n_mics-1:sub, :Nt_half:a_downsample]
    u = data['h'][:, :n_mics-1:sub, :Nt_half:u_downsample]
    x = data['x'][:, :n_mics-1:sub]

    # Skip times in u which is also in a
    inx = np.array([i for i in range(int(Nt_half/u_downsample)) if i%(targets_between_input+1)!=0])
    u = u[:, :, inx]

    return a, u, x


def create_input_tensor(a, x, targets_between_input=3, Nt=1201, fs=1200,
                        a_downsample=24):
    """
    From the simulated data form the input tensor A.

    Parameters
    ----------
    a : Torch Tensor, size=(n_sims, N_x, N_a)
        Input impulse response measurements.
    x : Torch Tensor, size=(n_sims, N_x)
        Spatial positions of microphones.
    targets_between_input : int, optional
        The number of target time locations between input time locations.
        The default is 3.
    Nt : int, optional
        Number of temporal sample points used in the simulation. The default is 1201.
    fs : int, optional
        The sample frequency. The default is 1200.
    a_downsample : int, optional
        Downsampling in the temporal axis for the available temporal
        sample points in the input. The default is 24.

    Returns
    -------
    A : Torch Tensor, size=(n_sims, N_x, N_u, N_a+2)
        Input tensor.

    """
    Nt_half = ceil((Nt)/2)
    n_sims, N_x, N_a = np.shape(a)
    N_u = int((N_a-1)*targets_between_input)

    u_downsample = ceil((a_downsample-1)/(targets_between_input + 1))

    # Repeat data to construct A_p
    a = a.reshape(n_sims, N_x, 1, N_a).repeat([1, 1, N_u, 1])

    # Construct the augmentation part of A
    gridx_train = x[:n_sims, :].reshape(n_sims, N_x, 1, 1).repeat([1, 1, N_u, 1])

    t_half_u = np.linspace(0, (Nt-1)/fs, Nt)[:Nt_half:u_downsample]
    inx = np.array([i for i in range(len(t_half_u)) if i%(targets_between_input+1)!=0])
    gridt = torch.tensor(t_half_u[inx], dtype=torch.float)
    gridt = gridt.reshape(1, 1, N_u, 1).repeat([1, N_x, 1, 1])

    # Construct tensor A
    A = torch.cat((gridx_train, gridt.repeat([n_sims, 1, 1, 1]), a), dim=-1)
    return A


if __name__ == '__main__':
    data_type = 'test'
    #data_type = 'validation'
    #data_type = 'train'


    ### Path to Data ###
    if data_type == 'test' or data_type == 'train':
        PATH = 'IR_'+data_type+'1000.mat'
    elif data_type == 'validation':
        PATH = 'IR_'+data_type+'400.mat'

    save_folder = data_type+'_data/'


    ### Form and Save Data Sets ###
    if data_type == 'test' or data_type == 'validation':
        for targets_between_input in [1, 2, 3, 5, 11, 23]:
            a, u, x = input_data(PATH, targets_between_input=targets_between_input)
            print(np.shape(a), np.shape(u))
            savemat(save_folder+data_type+'_u'+str(targets_between_input)+'.mat', {"u": u})
    elif data_type == 'train':
        for targets_between_input in [1, 2, 3]:
            a, u, x = input_data(PATH, targets_between_input=targets_between_input)
            print(np.shape(a), np.shape(u))
            savemat(save_folder+data_type+'_u'+str(targets_between_input)+'.mat', {"u": u})

    savemat(save_folder+data_type+'_a.mat', {"a": a})
    savemat(save_folder+data_type+'_x.mat', {"x": x})
