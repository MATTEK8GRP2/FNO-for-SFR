# -*- coding: utf-8 -*-
"""


Created on Mon Apr 19 11:42:52 2021


Authors:  Martin Voigt Vejling, Simone Birk Bols Thomsen and
          Kristian Søgaard
E-Mails: {mvejli17, sbbt17, ksagaa17}@student.aau.dk

In this module the relative $\ell_2$ error is implemented. This is used as the
loss both during training and testing as described in the report

        Fourier Neural Operators for Sound Field Reconstruction
                        in the Time Domain
           - Chapter 7: Temporal Sound Field Reconstruction
                - Chapter 8: Numerical Experiments

This module is used in the scripts train_FNO_network_2d.py,
validation_err_vs_res_plot.py, and test_FNO_network_2d.py.

The script has been developed using Python 3.6 with the library torch 1.7.0. 


"""

import torch


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        """
        Parameters
        ----------
        p : int, optional
            Order of the norm. The default is 2.
        size_average : boolean, optional
            Boolean variable to choose whether the loss should be averaged
            or summed across a batch. The default is True.
        reduction : boolean, optional
            Boolean variable to choose whether the loss should be returned as
            an array or a float. The default is True.
        """
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert p > 0

        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, U_hat, U):
        """
        The relative $\ell_p$ error for prediction U_hat and ground truth U.
        """
        num_examples = U_hat.size()[0]

        diff_norms = torch.norm(U_hat.reshape(num_examples,-1) - U.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(U.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

