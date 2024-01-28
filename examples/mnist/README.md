A simple CNN build on top of `tt.h` and `nn.h` and the MNIST training example. Currently no batch training.

`data.c` parse the raw MNIST dataset and generate `.tt` format tensor data files. `train.c` and `test.c` is for training and testing. See `Makefile`.