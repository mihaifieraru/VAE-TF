## Implementation of Variational Auto Encoder in Tensorflow

The model parameters (such as number of iterations, latent space size, train/test, number of generated images) can be changed either by changing the flags in the code or by setting them when running the command.
The following parameters can be set via flags:
	**DATA_DIR** - Directory for storing data
	**MODEL_PATH** -Path to the parameters of the trained model
	**INPUT_SIZE** - Size of the input image
	**HIDDEN_ENCODER_SIZE** - Size of the hidden layer in the encoder
	**HIDDEN_DECODER_SIZE** - Size of the hidden layer in the decoder
	**LATENT_SPACE_SIZE** - Size of the latent space
	**ADAGRAD_LR** - Learning rate Adagrad
	**MINIBATCH_SIZE** - Size of minibatch
	**NUMBER_ITERATIONS** - Number of iterations for optimization
	**INIT_STD_DEV** - Standard deviation for the truncated normal used for initializing the weights
	**TRAIN** - Specifies whether to train a model or to use a preexistent one
	**TEST_THE_TRAINING** - Specifies whether or not to do testing
	**GENERATE** - Specifies whether to generate images from noise or not
	**NUMBER_IMAGES_TEST_THE_TRAINING** - Number of images to show in tensorboard
	**NUMBER_IMAGES_GENERATED** - Number of images to generate from noise



The model will be stored in the model/ folder
Running test phase without a training phase is possible if a model already exists in the model/ folder

The log files will be stored in the logs/ folder

After running the program, vizualization of the log data is possible via tensorboard by specifying the path to the log file:
	tensorboard --logdir=logs
