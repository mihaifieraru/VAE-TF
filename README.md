## Implementation of the Variational Auto Encoder in Tensorflow

Requires TensorFlow r0.11


To run on MNIST dataset:
```
python vae.py
```


The model parameters (such as number of iterations, latent space size, train/test, number of generated images) can be changed via command line. For a description:
```
python vae.py --help
```


The model will be stored in the model/ folder.
Training not necessary if a model already exists in the model/ folder:
```
python vae.py --TRAIN=False
```


The log files (events) will be stored in the logs/ folder.
After running the program, vizualize the results (initial vs decoded images and images generated from noise) in tensorboard by reading the events in the logs/ folder:
```
tensorboard --logdir=logs
```
Open the generated URL in the browser and navigate to IMAGES. 


### License
```
The MIT License (MIT)
Copyright (c) 2016 Mihai Fieraru, Alina Dima

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```