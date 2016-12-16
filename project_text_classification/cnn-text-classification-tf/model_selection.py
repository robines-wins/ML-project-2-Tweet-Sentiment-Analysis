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

def get_l2_parameter(list_lambda,list_dropout_p):  
	# Parameters
	# ==================================================
	# Data loading params
	tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
	tf.flags.DEFINE_string("positive_data_file", "../twitter-datasets/train_pos.txt", "Data source for the positive data.")
	tf.flags.DEFINE_string("negative_data_file", "../twitter-datasets/train_neg.txt", "Data source for the positive data.")

	# Model Hyperparameters
	tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
	tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
	tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
	tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
	tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

	# Training parameters
	tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
	tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 1)")
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


	test_losses = np.empty([len(list_lambda),len(list_dropout_p)]) 
	test_accuracies = np.empty([len(list_lambda),len(list_dropout_p)]) 
	train_losses = np.empty([len(list_lambda),len(list_dropout_p)]) 
	train_accuracies = np.empty([len(list_lambda),len(list_dropout_p)]) 
	
	for i_lamb,lamb in enumerate(list_lambda):
		FLAGS.l2_reg_lambda = lamb
		for i_drop,p in enumerate(list_dropout_p):
			FLAGS.dropout_keep_prob = p
			FLAGS.evaluate_every = 200
			_ ,last_test_loss,last_test_accuracy,last_train_loss,last_train_accuracy = train(FLAGS,w2v,vocab_processor)
			test_losses[i_lamb,i_drop] = last_test_loss
			test_accuracies[i_lamb,i_drop]=last_test_accuracy
			train_losses[i_lamb,i_drop]=last_train_loss
			train_accuracies[i_lamb,i_drop]=last_train_accuracy

	print("We obtained : ")
	for i_lamb,lamb in enumerate(list_lambda):
		for i_drop,p in enumerate(list_dropout_p):
			print("With (Lambda,p) = ({:.5f},{:.2f}) \ttrain_loss = {:.8f}\ttrain_accuracy = {:.8f}\ttest_loss = {:.8f}\ttest_accuracy = \t{:.8f}".format(lamb,p,train_losses[i_lamb,i_drop],train_accuracies[i_lamb,i_drop],test_losses[i_lamb,i_drop],test_accuracies[i_lamb,i_drop]))

	indices = np.argmax(test_accuracies)
	l_index,p_index = (indices/len(test_accuracies[0]),indices%len(test_accuracies[0]))
	optimal_lambda = list_lambda[l_index]
	optimal_dropout = list_dropout_p[d_index]
	optimal_accuracy = test_accuracies[l_index,p_index]

	print("\n(Lambda,p) which maximizes the accuracy =  ({:.5f},{:.2f}) with acc = {} ".format(optimal_lambda,optimal_dropout,optimal_accuracy))

if __name__ == "__main__":
	list_lambda = np.logspace(-4,1,10)
	list_dropout_p = [0.5,0.6,0.7,0.8,0.9]
	get_l2_parameter(list_lambda,list_dropout_p)
