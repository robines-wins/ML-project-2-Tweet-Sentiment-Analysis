# Tweet Sentiment Analysis using a convolutional neural network
Project realised in December 2016, for the class of Machine Learning (taught by M.Jaggi and R.Urbanke) during our Master in Communication Systems at EPFL. 
By :
- H. Moreau 
- R. Solignac
- A. Vandebrouck

## Aim of the project
This project aimed to do a binary classification on the sentiment expressed in 10'000 tweets using a training set 
of 2'270'482 distinct tweets. It used a convolutional neural network implemented using tensor flow (under Apache License) on top of word reprensentation using word2vec provided by gensim library. It achieved a fairly good accuracy of approximately 86%.

## Implementation
### Requirements
To run this code it is necessary to have installed on the computing machine the following:
- Python 3
- Tensorflow > 0.8 (avaible ```$ pip install tensorflow```)
- Numpy (avaible ```$ pip install numpy```)
- gensim (avaible ```$ pip install gensim```)
- (optional) GPU support for tensorflow using CUDA 8.0 and CUDNN 5.1
- download the training and testing sets from [the original kaggle competition](https://inclass.kaggle.com/c/epfml-text)

Note that those files are considered by the programm to be put in ```../twitter-datasets/```

### Basic run to obtain the kaggle submission
To obtain the kaggle submission, the only file to be ran is ```run.py``` (Using a terminal enter the folder ```cnn-text-classification-tf``` and input ```python run.py```)

### Using the neural network model 
#### Training
Also using a bash terminal inside the folder ```cnn-text-classification-tf``` call ```python train.py```. To ease up the set up of parameters we used Flags. Those can be listed using ```python train.py --help```

#### Eval
Now that the network has been trained it can predict on some new set using ```python eval.py --eval_train --checkpoint_dir="<path to run>```. Note that the trained network is checkpointed in the ```./runs/``` folder. Each run is backuped in a folder named after the time at which it was trained

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
4. ```generate_w2v.py``` : used to generate our own word2vec using [gensim libary](https://radimrehurek.com/gensim/models/word2vec.html)
 
#### Neural Network
1. ```text_CNN.py``` : the convolutional neural network (ses an embedding layer, followed by a convolutional, max-pooling and softmax layer.)
2. ```train.py``` : contains only one method used to train the neural network
3. ```eval.py``` : contains only one method used to evaluate our train NN on a new input

#### Ressearch Tools
1. ```Baseline_model.ipynb``` : contains the code used to generate our baseline model (Logisitc regression using the same word representation than the neural network)

### External contributions
In addition to the previously cited library, the tensor flow implemention of the CNN and parts of the evaulation and training protocols come from https://github.com/dennybritz/cnn-text-classification-tf under license Apache
