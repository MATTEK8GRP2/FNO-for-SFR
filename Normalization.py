# -*- coding: utf-8 -*-
"""


Created on Fri Apr 23 08:55:21 2021


Authors:  Martin Voigt Vejling, Simone Birk Bols Thomsen and
          Kristian Søgaard
E-Mails: {mvejli17, sbbt17, ksagaa17}@student.aau.dk

In this script the constant used for normalizing the data is computed and the
class used for normlizing and un-normalizing the data is introduced. This is
done as described in the report

        Fourier Neural Operators for Sound Field Reconstruction
                        in the Time Domain
           - Chapter 7: Temporal Sound Field Reconstruction
                    
This functionality is used in the scripts train_FNO_network_2d.py,
validation_err_vs_res_plot.py, and test_FNO_network_2d.py.

The script has been developed using Python 3.6 with the
libraries numpy, scipy, and torch 1.7.0.


"""

from scipy.io import loadmat, savemat
import numpy as np
import torch


class Max_Normalization(object):
    """
    Class for normalizing the data by dividing with the maximum value 
    of the impulse responses in the training data.
    """
    def __init__(self, mymax):
        super(Max_Normalization, self).__init__()
        mymax = mymax.astype(np.float32)
        self.mymax = torch.from_numpy(mymax)

    def encode(self, x):
        """
        Normalize the data.
        """
        return x/self.mymax

    def decode(self, y):
        """
        Un-normalize the data.
        """
        return y*self.mymax

    def to(self, device):
        """
        Send the normlization constant to the device (CUDA or CPU).
        """
        self.mymax = self.mymax.to(device)
        
        


if __name__ == '__main__':
    n_sims = 1000
    data = loadmat('data/IR_train{}.mat'.format(n_sims))

    ### Compute and Save the Constant Used for Normalization ###
    h = data["h"]
    p_max = np.max(h)
    p_max_dict = {"p_max": p_max.astype(np.float32)}
    savemat('data/max_normalization{}.mat'.format(n_sims), p_max_dict)
