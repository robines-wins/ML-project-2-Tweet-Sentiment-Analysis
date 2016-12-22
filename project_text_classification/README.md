# Tweet Sentiment Analysis using a convolutional neural network
Project realised in December 2016, for the class of Machine Learning (taught by M.Jaggi and R.Urbanke) during our Master in Communication Systems at EPFL. 
By :
- H. Moreau 
- R. Solignac
- A. Vandebrouck

## Aim of the project
With the rapid growth of online data, text classification has become one of the key technique to analyse data automatically. We could think as an example on doing online polling for future election by extracting the sentiment of tweets concerning a candidate, or getting feedback from customer upon the release/presentation of a new product. This classifications are to be realised on larger and larger sets of data, and are therefore in the heart of research for more efficient models. 
**This project aims to do text classification at a sentence level. More specifically to analyse the sentiments enclosed in tweets (binary classification : positive or negative)**. To do so, a large data set of 2 270 482 distinct tweets is used as a training set (due to computation time, a smaller subset of 181 321 distinct tweets is also considered). The final goal being to predict on 10 000 tweets.

Text classification usually relies on features developed by humans, one of the strength of this model relies in the lack of need for any human developed features.

Although this project uses a clean data set where sentences have already been tokenized/cleaned and a few modifications would be necessary to apply it to a different set of data.

## Implementation
### Requirements
To run this code it is necessary to have installed on the computing machine the following:
- Python 3
- Tensorflow > 0.8 (avaible ```$ pip install tensorflow```)
- Numpy (avaible ```$ pip install numpy```)
- gensim (avaible ```$ pip install gensim```)
- (optional) GPU support for tensorflow using CUDA 8.0 and CUDNN 5.1
- download the training and testing sets from [the original kaggle competition](https://inclass.kaggle.com/c/epfml-text)

Note that those files are considered by the programm to be put in ```.
./twitter-datasets/```

### Basic run to obtain the kaggle submission
To obtain the kaggle submission, the only file to be ran is ```run.py``` (Using a terminal enter the folder ```scripts``` and input ```python run.py```). To be able to run the model, **we provide the zip of the file ([link to run.zip](https://www.dropbox.com/s/8p1rm0wwpfpckpm/runs.zip?dl=0))**. This run.zip must be put in the folder ```scripts/``` and left compressed.

### Using the neural network model 
Note that right now the network is configured to run with a w2v, one should set the FLAGS properly to use the model without a w2v.

#### Training
Also using a bash terminal inside the folder ```scripts``` call ```python train.py```. To ease up the set up of parameters we used Flags. Those can be listed using ```python train.py --help```

#### Eval
Now that the network has been trained it can predict on some new set using ```python eval.py --eval_train --checkpoint_dir="<path to run>"``` (where ```<path-to-run>``` should reflect the path to dir where the trained model has been checkpointed, it will then choose the latest run as trained model). Note that the trained network is checkpointed in the ```./runs/``` folder. Each run is backuped in a folder named after the time at which it was trained

#### Using the w2v model builder
To generate a w2v, one should modify in the ```generate_w2v.py``` the path (absolute or relative) to the training and testing set by modifying 
```
filelist=['../twitter-datasets/train_pos_full.txt','../twitter-datasets/train_neg_full.txt','../twitter-datasets/test_data.txt']
```
Then, the w2v_path should be set to ```'../tweetdatabase_word2vec'``` in both the ```train.py``` and ```eval.py```. One should be careful to be coherent between the word representation in the train and eval steps that should be the same, otherwise the result will be completely random (the neural net being trained on a word representation and evaluated using another word representation for the new input)

### Architecture
All script files are in ```scripts/```

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
1. ```Baseline_model.ipynb``` : contains a notebook used to generate our baseline model (Logisitc regression using the same word representation than the neural network)

### External contributions
In addition to the previously cited library, the tensor flow implemention of the CNN and parts of the evaulation and training protocols come from https://github.com/dennybritz/scripts under license Apache
