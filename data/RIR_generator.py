# -*- coding: utf-8 -*-
"""


Created on Mon Apr  5 15:53:20 2021


Authors:  Martin Voigt Vejling, Simone Birk Bols Thomsen and
          Kristian Søgaard
E-Mails: {mvejli17, sbbt17, ksagaa17}@student.aau.dk

This script contains functionality to simulate Room Impulse Responses using
the image source method for a one-dimensional setup in space. For theoretical
aspects of the content of this simulation algorithm you are referred to
the report

        Fourier Neural Operators for Sound Field Reconstruction
                        in the Time Domain
                   - Chapter 6: Simulation Study
                    
The simulated data made with this script is called in the script
input_data_generation in which the training, testing, and validation data
sets are created.

The script has been developed using Python 3.6 with the
libraries numpy and scipy.


"""

import numpy as np
import scipy.io as sio
from numpy.random import uniform


class simulate_1d_data():
    """ 
    Randomly generate room impulse responses for the 1-dimensional setup
    with 'n_mic' microphones in each room and 'n_ls' sources placed randomly
    in each room. The number of different rooms are given by 'n_rooms' and the
    dimension of the rooms L_x is sampled from a uniform distribution. The
    default is Unif(3.5, 10). The number of simulations per room is
    n_sims_pr_room.
    
    Use this method by creating a class object calling 
            simulate_1d_data(*args, **kwargs)
    giving the inputs to the algorithm. Then call the function RIR() on the
    class to simulate the impulse responses.
    """
    def __init__(self, n_mic=32, n_ls=1, n_rooms=25, n_sims_pr_room=40,
                 room_dims=[3.5, 10], Or=-1, fs=1200, Nt=1201, c=340, f_c=300,
                 source_restriction=0.01):
        """
        Parameters
        ----------
        n_mic : int, optional
            Number of microphones. The defualt is 32.
        n_ls : int, optional
            Number of sources in each room. The default is 1.
        n_rooms : int, optional
            Number of rooms. The default is 25.
        n_sims_pr_room : int, optional
            Number of simulations per room with varying source positions.
            The default is 40.
        room_dims : list, optional
            Upper and lower bound on the uniform distribution from which
            L_x is sampled. The default is [3.5, 10].
        Or : int, optional
            The reflection order. If it is -1 then the maximum
            reflection order is used. The default is -1.
        fs : float, optional
            Sample frequency. The default is 1200.
        Nt : int, optional
            Number of samples in time. The default is 1201.
        c : float, optional
            Thermodynamic speed of sound. The default is 340.
        f_c : float, optional
            Cut-off frequency of lowpass filter. The default is 300.
        source_restriction : float, optional
            Specifies the minimum distance from a source to a microphone
            in cm. If it is None or 0 then the minimum distance is
            infinitesimal, but still the sources cannot be
            positioned exactly on the microphones. The default is 0.01 (1 cm).
        """
        self.n_mic = n_mic
        self.n_ls = n_ls
        self.n_rooms = n_rooms
        self.n_sims_pr_room = n_sims_pr_room
        self.n_sims = self.n_rooms * self.n_sims_pr_room
        self.fs = fs
        self.Nt = Nt
        self.c = c
        self.Omega_c = 2*np.pi*f_c

        # Generate rooms
        room_dims_ = np.random.rand(n_rooms)*(room_dims[1]-room_dims[0]) + room_dims[0]
        self.Lx = np.repeat(room_dims_, n_sims_pr_room, axis=0)

        # Time discretization
        Lt = (self.Nt-1)/self.fs
        self.Dt = np.linspace(0, Lt, self.Nt)

        # Check is we have a restriction on source position
        if source_restriction == None or source_restriction == 0:
            restriction_on_source_pos = False
        else:
            restriction_on_source_pos = True

        # Place microphones and sources (dont want source on a mic)
        if restriction_on_source_pos is True:
            self.x = np.zeros((self.n_sims, n_mic))
            self.x_s0 = np.zeros((self.n_sims, n_ls))
            for room_idx, Lx in enumerate(self.Lx):
                mic_w_ep = np.linspace(0, Lx, n_mic+2, endpoint=True)
                intervals = [(mic_w_ep[i]+source_restriction, mic_w_ep[i+1]-source_restriction) for i in range(n_mic+1)]
                self.x[room_idx, :] = mic_w_ep[1:-1]
                self.x_s0[room_idx, :] = self.random_from_intervals(intervals)
        else:
            self.x = np.zeros((self.n_sims, n_mic))
            self.x_s0 = np.zeros((self.n_sims, n_ls))
            for room_idx, Lx in enumerate(self.Lx):
                self.x[room_idx, :] = np.linspace(0, Lx, n_mic+2, endpoint=True)[1:-1]
                place_source = True
                while place_source is True:
                    self.x_s0[room_idx, :] = np.random.rand(n_ls)*Lx
                    for source in self.x_s0[room_idx, :]:
                        if not source in self.x[room_idx, :]:
                            place_source = False
                        else:
                            place_source = True
                            break

        # Set reflection order
        if Or == -1:
            self.Or = self.max_Or()
        else:
            self.Or = Or
        print("Reflection order: {}".format(self.Or))
        print("Number of simulations: {}".format(self.n_sims))

    def random_from_intervals(self, intervals):
        """
        This functions is used to place sources if a source_restriction
        is specified.

        Parameters
        ----------
        intervals : list
            List of (start, end) tuples for intervals.

        Returns
        -------
        float
            Random number in the union of the intervals.
        """
        total_size = sum(end-start for start,end in intervals)
        n = uniform(total_size)
        for start, end in intervals:
            if n < end-start:
                return start + n
            n -= end-start

    def max_Or(self):
        """
        Criterion for the maximum reflection order Or if the input is Or=-1.
        The ratio between the power from the original source and the image
        sources in decibel are used with a 60 dB attenuation criterion.

        dB = 10*log(\frac{\sum_t h_0(x, t)**2}{\sum_t h_Or(x, t)**2}),
        for Or = 1, ...

        When dB goes below -60 dB then the iteration stops and return Or.
        h_0 is formed by considering the maximum distance between a source
        and a microphone. Moreover, the largest room given as input is used.

        The approximation
            dist_Or = max(Lx)*(Or-1)
        is used for the distance of the image source of order Or to the
        microphones.

        Returns
        -------
        Or : int
            Reflection order.
        """
        max_source_mic_dist = np.max(np.abs(np.array(self.x) - np.array(self.x_s0)))
        factor1 = 2*self.Omega_c/(np.sqrt(2*np.pi) * max_source_mic_dist)
        factor2 = self.Omega_c*(self.Dt - max_source_mic_dist/self.c)/np.pi
        h_0_max = factor1*np.sinc(factor2)
        h_0_max_power = np.mean(h_0_max**2)
        Lx_max = np.min(self.Lx)

        dB = 0
        Or = 1
        while dB > -60:
            Or += 1
            dist_Or = Lx_max*(Or - 1) # The images sources are guaranteed to be further away
            factor1 = 2*self.Omega_c/(np.sqrt(2*np.pi) * dist_Or)
            factor2 = self.Omega_c*(self.Dt - dist_Or/self.c)/np.pi
            h_Or = factor1*np.sinc(factor2)
            h_Or_power = np.mean(h_Or**2)
            dB = 10*np.log(h_Or_power/h_0_max_power)
        return Or

    def RIR(self):
        """
        Main function.

        Returns
        -------
        h : ndarray, size=(n_rooms, n_mic, Nt)
            Room impulse responses.
        self.x : ndarray, size=(n_rooms, n_mic)
            Microphone positions.
        self.x_s0 : ndarray, size=(n_rooms, n_ls)
            Source positions.
        self.Lx : ndarray, size=(n_rooms)
            Room dimensions.
        self.Or : int
            Reflection order.
        """
        h = np.zeros((self.n_sims, self.n_mic, self.Nt))
        for room_idx, Lx in enumerate(self.Lx):
            print('Room index ', room_idx)
            h[room_idx, :, :] = self.RIR_one_room(Lx, room_idx)
        return h, self.x, self.x_s0, self.Lx, self.Or

    def RIR_one_room(self, Lx, room_idx):
        """
        Computes the impulse response using the image source method
        for a single room with dimension Lx.
        See algorithm 2 (Impulse Response Generator).

        Parameters
        ----------
        Lx : float
            Room dimension.
        room_idx : int
            Room index used to find microphone and source positions.

        Returns
        -------
        h : ndarray, size=(n_mic, Nt)
            Impulse response in the discrete points defined by Dt and self.x.
        """
        x_s0 = self.x_s0[room_idx, :]
        x = self.x[room_idx, :]

        self.x_image = np.zeros((2*self.Or+1, self.n_ls)) #Image source locations
        h_j = np.zeros((self.n_mic, self.Nt, self.n_ls)) #Impulse response for each source

        for j in range(self.n_ls):
            self.x_image[0, j] = x_s0[j] # Husk at den originale source er i starten
            for i in range(self.Or):
                self.x_image[2*(i+1)-1, j] = -self.x_image[2*(i+1)-2, j] # x_{s_{-i}}
                self.x_image[2*(i+1), j] = Lx + np.abs(Lx - self.x_image[2*(i+1)-1, j]) # x_{s_i}

            h_j[:, :, j] = self.compute_terms_in_RIR_with_lowpass_filter(j, x)

        h = np.sum(h_j, axis=2) #Impulse response
        return h

    def compute_terms_in_RIR_with_lowpass_filter(self, j, x):
        """
        This method computes the impulse response for a single microphone
        position for one original source and is called in the function
        RIR_one_room() in which the contribution from this method is summed
        for all the different original sources.

        Parameters
        ----------
        j : int
            Source index.
        x : ndarray
            Microphone positions.

        Returns
        -------
        h_j_entry : ndarray, size=(n_mic, Nt)
            Entry in h_j in self.RIR() function.
        """
        h_j_entry = np.zeros((self.n_mic, self.Nt))
        for t_idx, t in enumerate(self.Dt):
            for mic_idx, mic in enumerate(x):
                distance = np.array([np.abs(mic-self.x_image[i, j]) for i in range(2*self.Or+1)])
                factor1 = 2*self.Omega_c/(2*np.pi*distance)
                factor2 = self.Omega_c*(t - distance/self.c)/np.pi
                h_j_entry[mic_idx, t_idx] = np.sum(factor1*np.sinc(factor2))
        return h_j_entry


if __name__ == '__main__':
    np.random.seed(0)

    ## Simulate data ###
    sim_RIR = simulate_1d_data(n_mic=32, n_ls=1, n_rooms=25, n_sims_pr_room=40,
                                room_dims=[3.5, 10], Or=0,
                                fs=1200, Nt=1201,
                                c=343, f_c=300/(2*np.pi),
                                source_restriction=0.01)

    h, x, x_s0, Lx, Or = sim_RIR.RIR()

    ### Save data ###
    dictionary = {'h': h, 'x': x, 'x_s0': x_s0, 'Lx': Lx, 'Or': Or}
    sio.savemat('IR_train1000.mat', dictionary)
