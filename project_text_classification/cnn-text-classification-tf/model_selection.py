import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
import vocabulary
import word2vec
from train import train

def get_l2_parameter(list_param):  
	# Parameters
	# ==================================================
	# Data loading params
	tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
	tf.flags.DEFINE_string("positive_data_file", "../twitter-datasets/train_pos.txt", "Data source for the positive data.")
	tf.flags.DEFINE_string("negative_data_file", "../twitter-datasets/train_neg.txt", "Data source for the positive data.")

	# Model Hyperparameters
	tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
	tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
	tf.flags.DEFINE_integer("num_filters", 300, "Number of filters per filter size (default: 128)")
	tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
	tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

	# Training parameters
	tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
	tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
	tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
	tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
	#tf.flags.DEFINE_boolean("use_w2v", False, "use precomputed word2vec vector")
	# Misc Parameters
	tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
	tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()

	w2v = word2vec.Word2vec()
	print("Loading data...")
	x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
	max_document_length = max([len(x.split(" ")) for x in x_text])
	vocab_processor = vocabulary.Vocabulary(max_document_length,w2v,FLAGS.embedding_dim)

	test_losses = []
	test_accuracies = []
	train_losses = []
	train_accuracies = []
	
	for param in list_param :
		FLAGS.l2_reg_lambda = param
		_ ,last_test_loss,last_test_accuracy,last_train_loss,last_train_accuracy = train(FLAGS,w2v,vocab_processor)
		test_losses.append(last_test_loss)
		test_accuracies.append(last_test_accuracy)
		train_losses.append(last_train_loss)
		train_accuracies.append(last_train_accuracy)

	optimal = np.argmax(test_accuracies)
	max_acc = np.max(test_accuracies)
	print("Parameter which maximizes the accuracy =  {} with acc = {} ")
	print("We obtained : ")
	for idx,param in list_param:
		print("With L2 regularizer = {:.5f}\n\ttrain_loss = {:.8f}\n\ttrain_accuracy = {:.8f}\n\ttest_loss = {:.8f}\n\ttest_accuracy = {:.8f}".format(param,train_losses[idx],train_accuracies[idx],test_losses[idx],test_accuracies[idx]))

if __name__ == "__main__":
	list_param = [0.5]
	get_l2_parameter(list_param)
