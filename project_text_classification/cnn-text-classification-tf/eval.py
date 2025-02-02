#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
import vocabulary
import word2vec
import csv

def eval(FLAGS, w2v = None):
    """
    Method to evaluate our model on a new input

    IN : 
    FLAGS :     the different parameters of the training (see below for further details)
    w2v :       the word2vec that are pretrained (Default : None, random vector case)
    """
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    if FLAGS.eval_train:
        x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
        y_test = np.argmax(y_test, axis=1)
        x_id = range(1,len(x_raw)+1)
    else:
        x_id, x_raw = data_helpers.load_data_eval(FLAGS.eval_data_file)
        y_test = None
        

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = vocabulary.Vocabulary.restore(vocab_path,w2v)
    x_test = np.array(vocab_processor.transform(x_raw))

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Format predictions
    all_predictions_f = [1 if i==1.0 else -1 for i in all_predictions]

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_id), all_predictions_f))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerow(['Id','Prediction'])
        csv.writer(f).writerows(predictions_human_readable)

if __name__ == '__main__':
    # Parameters
    # ==================================================

    # Data Parameters
    tf.flags.DEFINE_string("positive_data_file", "../twitter-datasets/train_pos.txt", "Data source for the positive data (training).")
    tf.flags.DEFINE_string("negative_data_file", "../twitter-datasets/train_neg.txt", "Data source for the negative data (training).")
    tf.flags.DEFINE_string("eval_data_file", "../twitter-datasets/test_data.txt", "Data source for the evaluation.")
    tf.flags.DEFINE_string("w2v_path", "", "path to the precomputed word2vec vector (Default: None)")


    # Eval Parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
    tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    w2v = word2vec.Word2vec(FLAGS.w2v_path) if FLAGS.w2v_path != "" else None #load the word2vec database, if path is empty use random vector
    eval(FLAGS,w2v)