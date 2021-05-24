# Fourier Neural Operators for Sound Field Reconstruction in the Time Domain

This GitHub repository contains code for the 8th semester project on Fourier Neural Operators for Sound Field Reconstruction in the Time Domain made by graduate students in Mathematical-Engineering at Aalborg University.

Authors:	Simone Birk Bols Thomsen, Kristian Søgaard, and Martin Voigt Vejling
E-Mails:	{sbbt17, ksagaa17, mvejli17}@student.aau.dk

In this work a Fourier neural operator is used for sound field reconstruction in the time domain. The experiments are made on simulated room impulse responses using the image source method. Using the simulated data an FNO and a baseline model is trained and tested for different spatio-temporal discretizations.

## Files
The code consists of a number of scripts and modules including dependencies between the scripts. The dependencies will be outline following a listing of the included scripts.

- 'Normalization.py' is used to create a normalizer used to normalize the impulse responses used during training and evaluation of the models.
- 'Loss_Function.py' is a module containing the loss function used for neural network training.
- 'train_FNO_network_2d.py' is the script used to train the neural networks.
- 'test_FNO_network_2d.py' is the script used to test the neural networks on the different test data sets.
- 'validation_err_vs_res_plot.py' is the script used to evaluate the neural networks on the different validation data sets.
- data/
	- 'RIR_generator.py' is used to simulate room impules responses.
	- 'input_data_generation.py' is used to split the simulated data into the used training, validation, and test sets as well as forming the input tensor to the neural networks.
	- test_data/ is the folder used to hold the test data sets.
	- train_data/ is the folder used to hold the training data sets.
	- validation_data/ is the folder used to hold the validation data sets.
- networks/
	- 'Baseline_Network.py' is the neural network used as a baseline model.
	- 'FNO_network_2d.py' is the FNO neural network.
- results/
	- Train/ is a folder to store results from training.
	- Test/ is a folder to store results from testing.
- model/ is a folder to store the trained models.

To use the code follow the steps as described below:

- Run the "RIR_generator.py" script to simulate room impulse responses used for training, validation, and test.
- Run the 'input_data_generation.py' script importing the previously simulated room impulse responses for the training, validation, and test data. In this way the data sets with different discretizations for training, validation, and testing is formed and save in their respective folders.
- Run the script 'Normalization.py' to save the constant used for normalization of the data.
- Run the training script 'train_FNO_network_2d.py' thereby creating a PyTorch model.
- Use the created PyTorch model to compute validation losses with the script 'validation_err_vs_res_plot.py'.
- Finally, to compute the loss on the test data use the script 'test_FNO_network_2d.py'.