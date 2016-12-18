# Tweet Sentiment Analysis using a convolutional neural network
Project realised in December 2016, for the class of Machine Learning, Master in Communication Systems at EPFL. 
By :
- Hugo Moreau 
- Robin Solignac
- Axel Vandebrouck

## Aim of the project
This project aimed to do a binary classification on the sentiment expressed in 10'000 tweets using a training set 
of 2'270'482 distinct tweets. It used a convolutional neural network implemented using tensor flow (under Apache License).

## Implementation
### Requirements
To run this code it is necessary to have installed on the computing machine the following:
- Python 3
- Tensorflow > 0.8
- Numpy
- (optional) GPU support for tensorflow using CUDA 8.0 and CUDNN 5.1
- download the training and testing sets from [the original kaggle competition](https://inclass.kaggle.com/c/epfml-text)

Note that those files are considered by the programm to be put in ```../twitter-datasets/```

### Architecture
#### Helpers
1. ```data_helpers.py``` : contains all the methods to load the data and do the preprocessing. It contains the following methods:
  * ```write```: Utility function used to be able to write some files to be able to check what some methods do (e.g. load_data_eval)
  *  ```delete_duplicate_lines``` : To delete the lines that are identical in the training data
  *  ```load_data_and_labels``` : Loads the training data, splits the data into words and generates labels.Returns split sentences and labels.
  *  ```load_data_eval``` :     Loads the data on wich we wish to evaluate the model, returns split sentences and labels.
  *  ```batch_iter``` :  Generates a batch iterator for a dataset.
2. ```word2vec.py``` : contains the methods related to use of a potential word2vec external library. (contains some utility functions)
3. ```vocabulary.py``` : contains all the methods used to generate our embedding matrix (sentence matrix representation). It contains the following methods:
  * ```init``` : Classic python initializer to construct the vocabulary object
  * ```fit``` : Builds the vocabulariy
  * ```transform``` : Will use the vocabulary built by fit(.) to transform the list of tweets into a list of vectors. Each vector corresponds to a sentence and contains the index in the embedding of the given word (obtained from the vocabulary
  * ```fit_transform````:  applies the two previous method and return the output of transform
  * ```embedding_matrix```: Getter for the embedding matrix (Transforms the embeding list into a np array)
  * extra utility functions
  
#### Neural Network
1. ```text_CNN.py``` : the convolutional neural network (ses an embedding layer, followed by a convolutional, max-pooling and softmax layer.)
2. ```train.py``` : contains only one method used to train the neural network
3. ```eval.py``` : contains only one method used to evaluate our train NN on a new input

#### Ressearch Tools
1. ```Baseline_model.ipynb``` : contains the code used to generate our baseline model (Logisitc regression using the same word representation than the neural network)
