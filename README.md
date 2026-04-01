# CIFAR-10 CNN Classifier

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify images from the CIFAR-10 dataset.

It uses data augmentation (random horizontal flips and crops), normalization, and a CNN with convolutional blocks, BatchNorm, ReLU, MaxPooling, and fully connected layers with Dropout. The model is trained with the Adam optimizer and CrossEntropyLoss, supporting GPU/Apple MPS acceleration.

Run the script to train the model and evaluate test accuracy. Training logs include per-epoch loss and accuracy, with the final test accuracy displayed at the end.
