# Vaccine-Sentiment-Classifier
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## Dataset
The dataset contains tweets about vaccines that are labeled 0 for neutral, 1 for anti-vax or 2 for pro-vax. In this repository I have developed vaccine sentiment classifiers using different models.

## Overview of the NLP models
### Logistic Regression (Softmax Regression)
* Preprocessing: None, Lematizing, Stemming
* Feature Extraction with vectorizers: Bag Of Words, TF-IDF, Hashing Vectorizer
* LogisticRegression: I tried to various values for some hyperparameters by hand and I found that probably the best parameters are solver=saga, C=1.0 and l2 or l1 for penalty. A grid search confirmed that.

### Feed Forward Neural Networks
* Loss Function: CrossEntropyLoss
* Activation Functions: ReLU, ELU, SELU, tanh, LeakyReLU
* Batch Normalization
* Dropout
* Batch size
* Feature Extraction: TF-IDF, GloVe

### Bidirectional Recursive Neural Networks with LSTM or GRU
I experimented with various parameters and ways of preprocessing. 
* Sequence lentgh: The RNNs take input in the form of [batch_size, sequence_length, input_size]. My first approach was to set a default sequence length and truncate or pad sequences to match that length. The second approach, which I tried aiming to get better scores was to create batches with variable sequnce length using BucketIterator. BucketIterator groups texts based on legth so that it minimizes padding. 
* Skip Connections: I implemented in a seperate class (for each sequence leght approach) RNN with skip connections based on the following structure.

![image](https://user-images.githubusercontent.com/60042402/167860747-bcb57f8d-d4ea-49c9-b599-94f7afdbc356.png)

For Hyperparameter Tuning I experimented with the following parameters:
* Learning rate
* Gradient Clipping
* Number of stacked RNNs 
* Hidden size
* Batch size
* Dropout
* Epochs
* Skip Connections
* LSTM vs GRU cells
* Attention Mechanism

### BERT (Bidirectional Encoder Representations for Transformers)
I used the pretrained BERT-base-uncased and fine tuned it for our classification problem. For hyperparameter tuning I tried mainly the ones that are suggested in BERT paper:
* Batch size:16, 32
* Learning Rate (Adam): 5e-5, 3e-5, 2e-5
* Number of epochs: 2, 3, 4
I found that for batch size 16, learning rate 2e-5 and 3 epochs I got the best results.

Some results:

![image](https://user-images.githubusercontent.com/60042402/167922374-70465e33-4573-4a00-8fbf-6fd90fe037d9.png)

### Results Summary 
![image](https://user-images.githubusercontent.com/60042402/167922140-0ec37f54-5e78-4b25-abef-396362e45dec.png)


