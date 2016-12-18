from train import train
from eval import eval
import generate_w2v
import word2vec
import tensorflow as tf

#external files parameter
tf.flags.DEFINE_string("positive_data_file", "../twitter-datasets/train_pos_full.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../twitter-datasets/train_neg_full.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("w2v_path", "../tweetdatabase_word2vec", "path to where word2vec comouted file will be stored ")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_data_file", "../twitter-datasets/test_data.txt", "Data source for the evaluation.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_float("dev_sample_percentage", .01, "Percentage of the training data to use for validation")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Eval Parameters
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

#generate word2vec model
w2vfilelist =['../twitter-datasets/train_pos_full.txt','../twitter-datasets/train_neg_full.txt','../twitter-datasets/test_data.txt']
generate_w2v.generate_word2vec(w2vfilelist,FLAGS.w2v_path,FLAGS.embedding_dim)
#wrap word2vec model
w2v = word2vec.Word2vec(FLAGS.w2v_path)
#train our CNN and get back the directory of it
FLAGS.checkpoint_dir,_,_ = train(FLAGS,w2v)
#eval our evalution set using our model
eval(FLAGS,w2v)