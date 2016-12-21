# File used to gridsearch our values, it should have been 
# done with a proper Cross validation but due to time constraints, we simply did it like this
from train import train
from eval import eval
import generate_w2v
import word2vec
import tensorflow as tf


#external files parameter
tf.flags.DEFINE_string("positive_data_file", "../twitter-datasets/train_pos.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../twitter-datasets/train_neg.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("w2v_path", "../tweetdatabase_word2vec", "path to where word2vec comouted file will be stored ")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_data_file", "../twitter-datasets/test_data.txt", "Data source for the evaluation.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 1)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 1)")
tf.flags.DEFINE_integer("evaluate_every", 2545, "Evaluate model on dev set after this many steps (default: 2545)")
tf.flags.DEFINE_integer("checkpoint_every", 2500, "Save model after this many steps (default: 2500)")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation (default : 1/10)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Eval Parameters
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Get the w2vv (the program will assume it has already been constructed and will complain if it hasn't)
w2v = word2vec.Word2vec(FLAGS.w2v_path)
num_filters=[64,128,200,300,400,500]
filter_sizes=["2","3","4","5","6","2,3,4","3,4,5","4,5,6","2,3,4,5","3,4,5,6"]

filter_sizes=["2,3,4","4,5,6"]
num_filters=[64,200,300,400,500]

for fs in filter_sizes:
	for nf in num_filters:
		FLAGS.filter_sizes = fs
		FLAGS.num_filters = nf
		_,loss,accuracy = train(FLAGS,w2v)
		s = str(FLAGS.num_filters)+" "+str(FLAGS.filter_sizes)+" "+str(loss)+" "+str(accuracy)+"\n"
		print(s) 
		f = open('GS.txt','a')
		f.write(s)
		f.close()

FLAGS.filter_sizes = "2,3,4,5"
num_filters=[128]

for nf in num_filters:
	FLAGS.num_filters = nf
	_,loss,accuracy = train(FLAGS,w2v)
	s = str(FLAGS.num_filters)+" "+str(FLAGS.filter_sizes)+" "+str(loss)+" "+str(accuracy)+"\n"
	print(s) 
	f = open('GS.txt','a')
	f.write(s)
	f.close()

