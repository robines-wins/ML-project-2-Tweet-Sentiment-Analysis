import numpy as np
import re
import itertools
from collections import Counter

def write(list_,name,path='.'):
    """
    Utility function used to be able to write some files to be able to check what some methods do (e.g. load_data_eval)

    IN : 
    list_:  the list of lines to write in the file
    name :  the file name
    path :  the path where we want to write de file (Default: in the current directory)
    """
    f = open(path+'/'+name,'w')
    f.write(str(type(list_[0])) + '\n')
    for s in list_:
        f.write(str(s) + '\n')
    f.close()

def delete_duplicate_lines(x_raw):
    """
    To delete the lines that are identical in the training data
    
    IN : 
    x_raw :  clean data (list of clean sentences)
    y : associated labels

    return x_raw without the duplicated lines
    """
    print("\nDeleting duplicates in data...")
    seen = set()
    unique_x = []
    for line in x_raw:
        if line not in seen:
            seen.add(line)
            unique_x.append(line)
    perc_duplicate = (len(x_raw)-len(unique_x))/(len(x_raw))*100
    print("Found : {}% duplicates in the input\n".format(perc_duplicate,len(x_raw)))
    return unique_x


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads the training data, splits the data into words and generates labels.
    Returns split sentences and labels.

    IN : 
    positive_data_file :    path to the positive data file
    negative_data_file :    path to the negative data file

    OUT : 
    returns the input data x_text and the output in corresponding order
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    positive_examples = delete_duplicate_lines(positive_examples)

    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    negative_examples = delete_duplicate_lines(negative_examples)

    # Split by words
    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_eval(eval_data_file):
   """
    Loads the data on wich we wish to evaluate the model, 
    Returns split sentences and labels.

    IN : 
    eval_data_file :        path to the data on which we wish to run our evaluation
    negative_data_file :    path to the negative data file
    check_write :           option to be able to write the output to a file to check the function well-behavior

    OUT : 
    split sentences along with their ids from the evaluation data
    """

    # Load data from files
    eval_example = list(open(eval_data_file, "r").readlines())
    eval_example = [s.strip() for s in eval_example]
    #split id and text
    eval_example = [s.split(',',1) for s in eval_example]
    x_id, x_text = zip(*eval_example)

    # (optionnal) write list on file for check
    #write(x_id,"x_id.txt")
    #write(x_text,"x_text.txt")

    # Split by words
    return x_id, x_text


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset. Will produce roughly (data_size/batch_size) * num_epochs batches

    IN : 
    data :          the zip of the input and ouput (each element is the list is the pair (x,y))
    batch size :    the number of (x,y) pairs per batch
    num_epochs :    the number of times we want to go through the full set of data
    shuffle :       each epoch through the full data set, if set to True we will use new random batches

    OUT : 
    yields a new batch 
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
