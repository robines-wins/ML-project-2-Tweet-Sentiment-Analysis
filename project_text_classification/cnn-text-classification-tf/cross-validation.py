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

def build_k_indices(y, k_fold, seed):
    """build k groups of indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold) #50 000
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(FLAGS,k_fold,x,y):
    """Estimates the test error of a given hyperparameter choice by doing cv on 5-fold"""
    
    # Printing the parameters
    print("\nParameters:")
    paraml = []
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
        paraml.append("{}={}".format(attr.upper(), value))
    print("")

    # Data Preparation
    # ==================================================
    seed = 3
    k_fold_indices = build_k_indices(y, k_fold, seed)

    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x.shape[1],
                num_classes=y.shape[1],
                vocab_size=vocab_processor.vocsize,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                embedding = vocab_processor.embeddingMatrix(),
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
           
            # Output directory for models and summaries
            timestamp =  time.strftime('%Y-%m-%d-%H-%M-%S') 
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            #train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            #write list of flags
            data_helpers.write(paraml,"train_flags.txt",str(out_dir))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step

                IN :
                x_batch :   input for the training step
                y_batch :   labels for the training step

                OUT : 
                Loss and accuracy of the step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
                return loss,accuracy

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set

                IN :
                x_batch :   input for the training step
                y_batch :   labels for the training step
                writer :    writer to save the different variables to be visualized in tensorboard

                OUT : 
                Loss and accuracy of the step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return loss,accuracy


            test_losses = []
            train_losses = []
            test_accuracies = []
            train_accuracies = []

            for k in k_fold:
                train_indices = k_indices[[i for i in range(len(k_indices)) if i != k]].ravel()
                test_indices = k_indices[k]

                x_train = x[train_indices]
                y_train = y[train_indices]
                x_test = x[test_indices]
                y_test = y[test_indices]

                # Generate batches
                train_batches = data_helpers.batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                test_batches = data_helpers.batch_iter(
                    list(zip(x_test, y_test)), FLAGS.batch_size, FLAGS.num_epochs)
                
                # Training loop. For each batch... 
                train_loss = 0
                train_accuracy = 0
                for batch in train_batches:
                    x_batch, y_batch = zip(*batch)
                    # We update the train loss and accuracy since as we go over the batch we keep training the NN
                    train_loss,train_accuracy = train_step(x_batch, y_batch)

                # Testing loop. For each batch...
                test_loss = 0
                test_accuracy = 0
                for batch in test_batches:
                    x_batch, y_batch = zip(*batch)
                    # We update the train loss and accuracy since as we go over the batch we keep training the NN
                    test_loss,test_accuracy = dev_step(x_batch, y_batch)

                # Add the loss and accuracy to the lists
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

            # Compute the mean over the fold
            mean_test_l = np.mean(test_losses)
            mean_train_l = np.mean(train_losses)
            mean_train_a = np.mean(train_accuracies)
            mean_test_a = np.mean(test_accuracies)
    return mean_test_l,mean_test_a,mean_train_l,mean_train_a


def get_best_hyper_parameters():
    # Parameters
    # ==================================================
    # Data loading params
    tf.flags.DEFINE_string("positive_data_file", "../twitter-datasets/train_pos.txt", "Data source for the positive data.")
    tf.flags.DEFINE_string("negative_data_file", "../twitter-datasets/train_neg.txt", "Data source for the positive data.")
    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 256 , "Batch Size ")
    tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 1)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    # Vocabulary 
    w2v = word2vec.Word2vec()

    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = vocabulary.Vocabulary(max_document_length,w2v,FLAGS.embedding_dim)
    x = np.array(vocab_processor.fit_transform(x_text))
    print("Vocabulary Size: {:d}".format(vocab_processor.vocsize))

    #Hyperparameters to test
    list_lambda = np.logspace(-4,1,10)
    list_dropout_p = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    test_losses = np.empty([len(list_lambda),len(list_dropout_p)]) 
    test_accuracies = np.empty([len(list_lambda),len(list_dropout_p)]) 
    train_losses = np.empty([len(list_lambda),len(list_dropout_p)]) 
    train_accuracies = np.empty([len(list_lambda),len(list_dropout_p)]) 
    
    for i_lamb,lamb in enumerate(list_lambda):
        FLAGS.l2_reg_lambda = lamb
        for i_drop,p in enumerate(list_dropout_p):
            FLAGS.dropout_keep_prob = p
            last_test_loss,last_test_accuracy,last_train_loss,last_train_accuracy = cross_validation(FLAGS,4,x,y)
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
    # Call get best
    get_best_hyper_parameters()
