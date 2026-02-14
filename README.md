# Deep Learning Notebooks Repository

This repository contains a collection of Jupyter notebooks covering fundamental and advanced topics in deep learning using Python, TensorFlow, and Keras. Each notebook is self-contained with explanations, code, and visualizations to help you understand key concepts.

## Directory Structure

- **`images/`** – Contains images used in the notebooks (e.g., diagrams, plots).
- **`model_ckps/`** – Directory for saving model checkpoints during training.
- **`README.md`** – This file.
- **`.ipynb` files** – Jupyter notebooks listed below.

---

## Notebooks

### 1. `BackProp.ipynb`
**Topic:** Backpropagation Algorithm  
**Description:**  
A step-by-step implementation of the backpropagation algorithm from scratch. It explains the mathematics behind gradient computation and weight updates in a neural network. Learn how forward and backward passes work, derive gradients for a simple feedforward network, manually implement backpropagation using NumPy, and compare with automatic differentiation in TensorFlow/Keras.

### 2. `autoencoders.ipynb`
**Topic:** Autoencoders  
**Description:**  
Introduction to autoencoders for unsupervised learning. Covers undercomplete and denoising autoencoders, building and training an autoencoder using Keras, visualizing latent representations and reconstructed images, and applications in dimensionality reduction and anomaly detection.

### 3. `cnn_with_keras.ipynb`
**Topic:** Convolutional Neural Networks (CNNs) with Keras  
**Description:**  
Demonstrates how to build and train CNNs for image classification using the Keras API. Includes convolutional layers, pooling, activation functions, data preprocessing and augmentation, training on datasets like MNIST or CIFAR-10, and evaluating model performance while visualizing filters.

### 4. `convolutional_neural_networks.ipynb`
**Topic:** Convolutional Neural Networks (In-depth)  
**Description:**  
A detailed exploration of CNNs, covering theoretical foundations of convolution, implementing a CNN from scratch using TensorFlow’s low-level APIs, advanced topics like stride, padding, dilation, modern architectures (e.g., ResNet blocks), and transfer learning with pre-trained models.

### 5. `introduction_to_artificial_neural_networks.ipynb`
**Topic:** Introduction to Artificial Neural Networks  
**Description:**  
A beginner-friendly notebook covering the basics of neural networks: perceptron model, activation functions, building a simple multi-layer perceptron (MLP) for classification, understanding loss functions and optimizers, and practical training tips (learning rate, batch size, etc.).

### 6. `recurrent_neural_networks.ipynb`
**Topic:** Recurrent Neural Networks (RNNs)  
**Description:**  
Introduces RNNs for sequence data. Topics include processing sequential data (time series, text), vanilla RNN cells and vanishing gradient issues, building an RNN for text generation or sentiment analysis, and visualizing hidden states and predictions.

### 7. `reinforcement_learning.ipynb`
**Topic:** Reinforcement Learning  
**Description:**  
Hands-on introduction to reinforcement learning concepts: Markov Decision Processes (MDPs) and Q-learning, implementing a simple agent in a grid-world environment, using neural networks as function approximators (Deep Q-Networks), and training an agent to play a game using OpenAI Gym.

### 8. `rnn_keras.ipynb`
**Topic:** RNNs with Keras  
**Description:**  
Focuses on implementing recurrent models using the Keras high-level API: building LSTM and GRU layers for sequence prediction, time-series forecasting (e.g., stock prices or weather data), handling variable-length sequences with masking, and saving/loading trained RNN models.

### 9. `tensorflow.ipynb`
**Topic:** TensorFlow Basics  
**Description:**  
An introduction to TensorFlow 2.x fundamentals: tensors, operations, eager execution, automatic differentiation with `tf.GradientTape`, building custom models and training loops, and using TensorBoard for visualization.

### 10. `tensorflow_keras_regression.ipynb`
**Topic:** Regression with TensorFlow/Keras  
**Description:**  
Focuses on regression tasks using neural networks: preparing data for regression (scaling, splitting), building a regression model with Keras, choosing loss functions (MSE, MAE) and metrics, and hyperparameter tuning and overfitting prevention.

### 11. `training_deep_neural_nets.ipynb`
**Topic:** Training Deep Neural Networks  
**Description:**  
Practical techniques for successfully training deep networks: weight initialization strategies, batch normalization and dropout, learning rate schedules and adaptive optimizers (Adam, RMSprop), and debugging gradient issues while monitoring training curves.

---

