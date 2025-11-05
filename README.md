# ml_genn_examples
Example scripts using the ml_genn machine learning framework for SNNs

## latency_mnist_conv
This is a script that was used to tune up a simple example of using convolutions and the eventprop compiler to learn the latency-encoded MNIST dataset. The script offers some diagnostic plots of the learned kernels in the convolutional layer.
- ```latency_mnist_conv.py```: Main script to train a convolutional neural network with one convolutional layer and dense head.
- ```plot_diagnostics.py```: Helper script to plot the evolution of spiking statistics in the hidden layer and training/validation accuracy across epochs using the output from the main script.
- ```plot_g.py```: Helper script to plot kernels (before training, after training and the difference)
