# Vaccine-Sentiment-Classifier
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## Dataset
The dataset contains tweets about vaccines that are labeled 0 for neutral, 1 for anti-vax or 2 for pro-vax. In this repository I have developed vaccine sentiment classifiers using different models.

## Models
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

### Bidirectional Recursive Neural Networks

### BERT (Bidirectional Encoder Representations for Transformers)
