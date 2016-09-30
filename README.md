## Implementation of Variational Auto Encoder in Tensorflow

The model parameters (such as number of iterations, latent space size, train/test, number of generated images) can be changed either by changing the flags in the code or by setting them when running the command.

The following parameters can be set via flags:

	__DATA_DIR__ - Directory for storing data

	__MODEL_PATH__ -Path to the parameters of the trained model

	__INPUT_SIZE__ - Size of the input image

	__HIDDEN_ENCODER_SIZE__ - Size of the hidden layer in the encoder

	__HIDDEN_DECODER_SIZE__ - Size of the hidden layer in the decoder

	__LATENT_SPACE_SIZE__ - Size of the latent space

	__ADAGRAD_LR__ - Learning rate Adagrad

	__MINIBATCH_SIZE__ - Size of minibatch

	__NUMBER_ITERATIONS__ - Number of iterations for optimization

	__INIT_STD_DEV__ - Standard deviation for the truncated normal used for initializing the weights

	__TRAIN__ - Specifies whether to train a model or to use a preexistent one

	__TEST_THE_TRAINING__ - Specifies whether or not to do testing

	__GENERATE__ - Specifies whether to generate images from noise or not

	__NUMBER_IMAGES_TEST_THE_TRAINING__ - Number of images to show in tensorboard

	__NUMBER_IMAGES_GENERATED__ - Number of images to generate from noise



The model will be stored in the model/ folder

Running test phase without a training phase is possible if a model already exists in the model/ folder

The log files will be stored in the logs/ folder

After running the program, vizualization of the log data is possible via tensorboard by specifying the path to the log file:
```
	tensorboard --logdir=logs
```
## Implementation of Variational Auto Encoder in Tensorflow

The model parameters (such as number of iterations, latent space size, train/test, number of generated images) can be changed either by changing the flags in the code or by setting them when running the command.

The following parameters can be set via flags:

	__DATA_DIR__ - Directory for storing data

	__MODEL_PATH__ -Path to the parameters of the trained model

	__INPUT_SIZE__ - Size of the input image

	__HIDDEN_ENCODER_SIZE__ - Size of the hidden layer in the encoder

	__HIDDEN_DECODER_SIZE__ - Size of the hidden layer in the decoder

	__LATENT_SPACE_SIZE__ - Size of the latent space

	__ADAGRAD_LR__ - Learning rate Adagrad

	__MINIBATCH_SIZE__ - Size of minibatch

	__NUMBER_ITERATIONS__ - Number of iterations for optimization

	__INIT_STD_DEV__ - Standard deviation for the truncated normal used for initializing the weights

	__TRAIN__ - Specifies whether to train a model or to use a preexistent one

	__TEST_THE_TRAINING__ - Specifies whether or not to do testing

	__GENERATE__ - Specifies whether to generate images from noise or not

	__NUMBER_IMAGES_TEST_THE_TRAINING__ - Number of images to show in tensorboard

	__NUMBER_IMAGES_GENERATED__ - Number of images to generate from noise



The model will be stored in the model/ folder

Running test phase without a training phase is possible if a model already exists in the model/ folder

The log files will be stored in the logs/ folder

After running the program, vizualization of the log data is possible via tensorboard by specifying the path to the log file:
```
	tensorboard --logdir=logs
```
