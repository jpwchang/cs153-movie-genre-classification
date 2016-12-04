import tensorflow as tf
import numpy as np
import argparse
import os.path
from glob import iglob, glob
from scipy.misc import imread
from random import shuffle
from sklearn.metrics import confusion_matrix, classification_report

def batch_generator(files, batch_size, num_labels):
    """
    Generator for on-demand loading of batch image data, so that all the images
    don't need to be loaded in memory at once. When all files in the given list
    are exhausted, the list is shuffled for reuse.
    """

    num_batches = len(files) // batch_size # how many batches we can go through before having to reshuffle
    cur_batch = 0
    while True:
        # reshuffle if we have run out of files
        if cur_batch == num_batches:
            shuffle(files)
            cur_batch = 0

        imgs = []
        labels = np.zeros([batch_size, num_labels])
        batch_files = files[cur_batch*batch_size:cur_batch*batch_size+batch_size]
        label_idx = 0
        for f in batch_files:
            imgs.append(imread(f[0], mode='RGB'))
            labels[label_idx, f[1]] = 1
            label_idx += 1
        images = np.stack(imgs, 0)

        cur_batch += 1

        yield images, labels

def verify_image(img_path, expected_size):
    """
    Helper function for filtering out broken images (e.g. download errors)
    """
    try:
        im = imread(img_path)
        return len(im.shape) > 0 and im.shape[0] == expected_size[0] and im.shape[1] == expected_size[1]
    except:
        return False

class PredictionModel(object):
    """
    Defines the machine learning model
    """

    def __init__(self, sess, training_data, image_shape, num_classes, step_size, batch_size):

        # save training data and associated information
        self.image_shape = image_shape
        self.num_classes = num_classes
        # generator for training batches
        self.batch_gen = batch_generator(training_data, batch_size, num_classes)

        self.data_size = len(training_data)
        self.step_size = step_size
        self.batch_size = batch_size

        # initialize TensorFlow graph variables
        self.X = tf.placeholder(tf.float32, shape=[None] + self.image_shape)
        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # finally, we need a TensorFlow session to run the model
        self.sess = sess

        self.build_model()

        # create a Saver instance for saving and loading the model
        self.saver = tf.train.Saver()

    def build_model(self):
        """
        Construct the TensorFlow computation graph
        """

        # we will do three convolutional layers, with max pooling at each
        # layer, so that the final images are 1/8 the size of the original
        h_conv1 = tf.contrib.layers.convolution2d(self.X, 64, 5)
        h_pool1 = tf.contrib.layers.max_pool2d(h_conv1, 2, 2)
        h_norm1 = tf.nn.local_response_normalization(h_pool1)

        h_conv2 = tf.contrib.layers.convolution2d(h_norm1, 128, 5, biases_initializer=tf.constant_initializer(0.1))
        h_pool2 = tf.contrib.layers.max_pool2d(h_conv2, 2, 2)
        h_norm2 = tf.nn.local_response_normalization(h_pool2)

        h_conv3 = tf.contrib.layers.convolution2d(h_norm2, 256, 5, biases_initializer=tf.constant_initializer(0.1))
        h_pool3 = tf.contrib.layers.max_pool2d(h_conv3, 2, 2)
        h_norm3 = tf.nn.local_response_normalization(h_pool3)

        # now we add two linear layers to do the prediction
        # first we need to flatten the convolution output
        h_conv_out = tf.contrib.layers.flatten(h_norm3)
#        h_linear1 = tf.contrib.layers.fully_connected(h_conv_out, 500,
#                                                      activation_fn=None,
#                                                      biases_initializer=tf.constant_initializer(0.1))
        # insert dropout to prevent overfitting
#        h_dropout = tf.nn.dropout(h_linear1, self.keep_prob)
        self.logits = tf.contrib.layers.fully_connected(h_conv_out, self.num_classes,
                                                        activation_fn=None,
                                                        biases_initializer=tf.constant_initializer(0.1))
        self.predictions = tf.nn.softmax(self.logits)
        self.logits = tf.Print(self.logits, [self.predictions], summarize=12)

        # since this is a classification task, we train on cross-entropy loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.Y))

    def train(self, epochs):
        """
        Main training loop
        """

        tf.global_variables_initializer().run()

        train_step = tf.train.GradientDescentOptimizer(self.step_size).minimize(self.loss)

        for epoch in range(epochs):
            # compute how many batches we can get from the data
            num_batches = self.data_size // self.batch_size
            for batch in range(num_batches):
                batch_x, batch_y = next(self.batch_gen)
                feed_dict = {self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.5}
                train_step.run(feed_dict=feed_dict)
                cur_loss = self.loss.eval(feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1.0})
                print("[Epoch %d | Batch %d/%d] Loss=%f" % (epoch+1, batch+1, num_batches, cur_loss))

        # save the trained model in case we want to reuse it later
        self.save()

    def test(self, test_files):
        """
        Evaluate the trained model on a held-out test set
        """

        test_batch_gen = batch_generator(test_files, self.batch_size, self.num_classes)
        # compute number of batches needed to get through the test data
        num_batches = len(test_files) // self.batch_size
        # arrays to hold the true and predicted labels
        y_true = np.zeros([num_batches * self.batch_size])
        y_pred = np.zeros([num_batches * self.batch_size])

        # run the model
        for batch in range(num_batches):
            batch_x, batch_y = next(test_batch_gen)
            feed_dict = {self.X: batch_x, self.Y: batch_y, self.keep_prob: 1.0}
            predictions, cur_loss = self.sess.run([self.predictions, self.loss], feed_dict=feed_dict)
            y_true[batch*self.batch_size:batch*self.batch_size+self.batch_size] = np.argmax(batch_y, 1)
            y_pred[batch*self.batch_size:batch*self.batch_size+self.batch_size] = np.argmax(predictions, 1)
            print("[TEST | Batch %d/%d] Loss=%f" % (batch+1, num_batches, cur_loss))

        # compute performance metrics
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))

    def save(self):
        self.saver.save(self.sess, "trained_model")

    def load(self):
        ckpt = tf.train.get_checkpoint_state('.')
        print(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))

def main():
    #labels = ["Action", "Comedy", "Documentary", "Horror", "Western"]
    labels = ["Action", "Comedy", "Horror", "Western"]

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Runs a machine learning model for automatic movie poster genre classification")
    parser.add_argument('--load', action='store_true',
                        help="Load a pre-trained model from disk instead of training from scratch")
    args = parser.parse_args()

    # load the image data
    images = []
    for label in labels:
        for img in glob("data_new/%s/*.jpg" % label)[:2900]:
            if verify_image(img, [192,128]):
                images.append((img, labels.index(label)))

    # make sure the labels are normally distributed
    shuffle(images)
    # use 1/4 of the data for the held-out test set
    num_train = int(0.75 * len(images))
    train_files = images[:num_train]
    test_files = images[num_train:]
    print("Training on %d images, testing on %d images" % (len(train_files), len(test_files)))

    with tf.Session() as sess:
        # create the model
        model = PredictionModel(sess, train_files, [192, 128, 3], len(labels), 1e-2, 50)
        if args.load:
            # load pre-trained model from disk
            model.load()
        else:
            # train the model
            model.train(3)
        model.test(test_files)

if __name__ == '__main__':
    main()
