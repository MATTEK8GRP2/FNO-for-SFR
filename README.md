# Fourier Neural Operators for Sound Field Reconstruction in the Time Domain

This GitHub repository contains code for the 8th semester project on Fourier Neural Operators for Sound Field Reconstruction in the Time Domain made by graduate students in Mathematical-Engineering at Aalborg University.

Authors:	Simone Birk Bols Thomsen, Kristian Søgaard, and Martin Voigt Vejling
E-Mails:	{sbbt17, ksagaa17, mvejli17}@student.aau.dk

In this work a Fourier neural operator is used for sound field reconstruction in the time domain. The experiments are made on simulated room impulse responses using the image source method. Using the simulated data an FNO and a baseline model is trained and tested for different spatio-temporal discretizations.

## Files
The code consists of a number of scripts and modules including dependencies between the scripts. The dependencies will be outline following a listing of the included scripts.

Markup : 	- 'Normalization.py' is used to create a normalizer used to normalize the impulse responses used during training and evaluation of the models.
		- 'Loss_Function.py' is a module containing the loss function used for neural network training.
		- 'train_FNO_network_2d.py' is the script used to train the neural networks.
		- data/
			- 'RIR_generator.py' is used to simulate room impules responses.
			- 'input_data_generation.py' is used to split the simulated data into the used training, validation, and test sets as well as forming the input tensor to the neural networks.
		- networks/
			- 'Baseline_Network.py' is the neural network used as a baseline model.
			- 'FNO_network_2d.py' is the FNO neural network.