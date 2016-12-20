# File used to reproduce the Kaggle score

from eval import eval
import tensorflow as tf
import zipfile

#external files parameter
tf.flags.DEFINE_string("checkpoint_dir", "./runs/2016-12-18-22-08-04/checkpoints", "Checkpoint directory from training run")
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
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 500)")
tf.flags.DEFINE_float("dev_sample_percentage", 0, "Percentage of the training data to use for validation (Default:0)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Eval Parameters
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Unizips the pretrained model
print("\nunzip trained model\n")
with zipfile.ZipFile("runs.zip","r") as zipf:
	zipf.extractall("")
print("\nDone !\n")

#eval our evaluation set using our model, we only need our vocabulary object, not anymore the word2vec database
eval(FLAGS)