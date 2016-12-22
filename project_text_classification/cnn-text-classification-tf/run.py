# File used to reproduce the Kaggle score

from eval import eval
import tensorflow as tf
import zipfile

#external files parameter
tf.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_data_file", "../twitter-datasets/test_data.txt", "Data source for the evaluation.")

# Eval parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Eval Parameters
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Unizips the pretrained model
try:
	print("\nunzip trained model\n")
	with zipfile.ZipFile("runs.zip","r") as zipf:
		zipf.extractall("")
	print("\nDone !\n")
	#eval our evaluation set using our model, we only need our vocabulary object, not anymore the word2vec database
	eval(FLAGS)
except FileNotFoundError as e:
	print(e)
	print("trained CNN archive not found. please put the run.zip archive from AAAAAAAAAAAAAAAAA in the same folder as this script and run this script again")