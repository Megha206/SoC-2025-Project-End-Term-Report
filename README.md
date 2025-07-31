# SoC-2025-Project-End-Term-Report
This repository contains the final project that I made over the course of SoC 2025, along with all my learnings

The second half of my SoC consisted of learning about Machine learning and Neural Networks. I learnt how to create a simple ML model using PyTorch. In this process, I understood what a PyTorch workflow looks like and implemented around 2-3 different models that did various things. 
I followed the PyTorch workflow of getting data ready, building/training a model, fitting the model to a dataset, making predictions with the model, evaluating the model, and saving and reloading the already trained model.

I implemented two main types of models:
1. Simple Linear Regression 
2. Binary Classification model
3. Along with these two main things, I did the exercises as suggested after each Notebook. This helped me understand the architecture of an ML model better.
   
I created a simple binary classification model and trained it using standard techniques for my project. 

# What I Learned
1. How to use torch.Tensor and the difference between NumPy and PyTorch tensors.
2. Model building using NN.Module, defining forward() methods, and compiling models with optimizers and loss functions.
3. Effective training loop structure: including loss tracking, model.train() vs model.eval() modes, and torch.no_grad() for inference.
4. Visualization of model performance using decision boundaries and accuracy/loss plots.
5. Building and training a neural network to classify points in a synthetic 2D circular dataset.
6. Using helper functions to reduce boilerplate and modularize training/evaluation logic.

# Challenges and Pain Points
1. Initially, grasping the training loop was tricky, especially understanding backward (), step (), and zeroing gradients.
2. Debugging shape mismatches during forward passes and loss computation required careful attention.
3. Choosing an appropriate model architecture and tuning the number of epochs and learning rate for convergence.
4. Grasping the math behind regression and other statistical concepts because I was unfamiliar with statistics, but this improved as I spent more time on the notebooks and exercises. 

 # Project Summary
This project was my way of implementing a series of deep learning models using PyTorch to perform binary and multi-class classification on hand-generated datasets such as concentric circles and Gaussian patterns. I started with basic feedforward architectures, exploring progressive model enhancements like adding hidden layers, activation functions (ReLU), and varying loss functions (binary cross-entropy, L1, cross-entropy), to improve classification performance.

## Key components include:
1. Synthetic Data generation  using inbuilt methods to generate data.
2. Model definitions via nn.Module and nn.Sequential, with explicit implementation of forward passes.
3. Training loops that manually handle forward propagation, loss computation, backpropagation, and optimization.
4. Evaluation using sigmoid/logit transformations for binary outputs and softmax for multi-class predictions.
5. Decision boundary visualizations and accuracy tracking across training and testing phases.
6. The final model achieves highly accurate nonlinear classification using deeper architectures and nonlinearities.
