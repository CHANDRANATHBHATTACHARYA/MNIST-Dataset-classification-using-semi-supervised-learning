Overview:
1)FASHION MNIST Classification with the help of Logistic Regression taking the help of Semi-Supervised learning
    This project aims to classify images from the Fashion MNIST dataset using a semi-supervised learning approach.
    Semi-supervised learning combines a small amount of labeled data with a large amount of unlabeled data to improve learning accuracy.
  
  Dataset:
    The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories, with 7,000 images per category.
    Each image is 28x28 pixels. The dataset is split into 60,000 training images and 10,000 test images.
  
  Methodology:
    .Data Preprocessing
       Normalization: Each image is normalized to have pixel values between 0 and 1.
       Reshaping: The images are reshaped to a 2D array of size 784 (28x28).
  
      K-means Clustering:
        Initialization: The K-means algorithm is applied to the unlabeled data to cluster similar images.
        Clustering: Clustering is performed with a predefined number of clusters (equal to the number of classes).
      
      Label Propagation
        Propagation: Labels are propagated from a subset of labeled data points to the unlabeled points based on their cluster assignments.
      
      Logistic Regression
        Training: A logistic regression model is trained on the combined dataset of labeled and pseudo-labeled data.
        Evaluation: The model is evaluated on the test set for accuracy and other performance metrics.
      
      Results
        The model's performance is evaluated using labelling of various amount of data using semi supervised learning. 
        The results show the effectiveness of the semi-supervised learning approach in improving the classification  accuracy of the Fashion MNIST dataset.

2)Overhead MNIST Classification with Neural Networks taking the help of Semi-Supervised Learning
      This repository contains a ipynb file that demonstrates the classification of the Overhead MNIST dataset using a neural network model. 
      The neural network is trained on the dataset to accurately classify the overhead images.

  Dataset:
      The Overhead MNIST dataset consists of grayscale images in 10 categories, with a predefined number of images per category. 
      Each image is 28x28 pixels. The dataset is split into training and test sets.

 Methodology:
   Data Preprocessing:
     Normalization: Each image is normalized to have pixel values between 0 and 1.
     Reshaping: The images are reshaped to a 2D array of size 784 (28x28) for input into the neural network.
     
   Neural Network Architecture:
       Input Layer: Accepts the 784-dimensional input vector.
       Hidden Layers: Multiple hidden layers with a specified number of neurons and activation functions (ReLU and softmax).
       Output Layer: A softmax layer to output probabilities for each of the 10 classes.
   Training:
      Loss Function: Sparse Categorical cross-entropy loss.
      Optimizer: Adam optimizer.
      Epochs: A predefined number of epochs for training.
      Batch Size: The number of samples per gradient update.
   Evaluation:
     Metrics: Accuracy is used to evaluate the model's performance on the test set.  
   Results:
     The model's performance is evaluated using various metrics. 
     The results demonstrate the neural network's ability to accurately classify the Overhead MNIST dataset.
     
      


