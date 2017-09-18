
import tensorflow as tf
import argparse
import numpy as np
import time

from com.mrt.audio.medleydb_analyser import MedleyDBAnalyser

N_BINS  =513
N_SAMPLES=20
VECTOR_SIZE=N_BINS * N_SAMPLES

logs_path = "/path/to/tensorboard/log/"
rootDir = '/path/to/MedleyDB/Audio/'
destDir = '/Volumes/Bhrigu/data/medleyDb/masks/'

class VocalSeparator:

    def __init__(self):
        print('init VocalSeparator... ')


    def variable_summaries(self,var):
        pass

    def _weightVariable(self,shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _biasVariable(self,shape,constVal=0.1):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.constant(constVal, shape=shape)
        return tf.Variable(initial)

    def buildModel(self):
        # Three layer fully connected model
        # input layer x
        # Hidden layer y
        # Output layer z
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, VECTOR_SIZE], name="x-input")
            z_ = tf.placeholder(tf.float32, shape=[None, VECTOR_SIZE], name="z-output")
            self.x = x
            self.z_ = z_
        with tf.name_scope("weights"):
            Wxy = self._weightVariable([VECTOR_SIZE, VECTOR_SIZE])
            # self.variable_summaries(Wxy)
            Wyz = self._weightVariable([VECTOR_SIZE, VECTOR_SIZE])
            # self.variable_summaries(Wyz)
        with tf.name_scope("biases"):
            by = self._biasVariable([VECTOR_SIZE],constVal=0.1)
            # self.variable_summaries(by)
            bz = self._biasVariable([VECTOR_SIZE],constVal=0.0)
            # self.variable_summaries(bz)
        with tf.name_scope("model"):
            # z is the prediction
            # y is the hidden layer
            # y = tf.nn.sigmoid(tf.matmul(x, Wxy) + by);
            y = tf.nn.sigmoid(tf.matmul(x, Wxy) + by,"hidden")

            # self._keep_prob = tf.placeholder(tf.float32)
            # y_drop = tf.nn.dropout(y, self._keep_prob)
            z  = tf.matmul(y, Wyz)
            self.z = z
        with tf.name_scope('cross_entropy'):
            # Cost 1
            # self._cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))

            # Cost 2 , alternative
            self._cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=z_, logits=z))

        with tf.name_scope('Accuracy'):
            # Accuracy = dot product between perdiction and ideal mask
            p = tf.sigmoid(z)
            p = tf.round(p)

            dotProduct = tf.reduce_sum(tf.multiply(z_, p), 1)
            norm_z = tf.sqrt(tf.reduce_sum(tf.multiply(p, p), 1))
            norm_z_ = tf.sqrt(tf.reduce_sum(tf.multiply(z_, z_), 1))
            norm = tf.multiply(norm_z, norm_z_)
            dotProduct = tf.divide(dotProduct, norm)
            self._accuracy_1 = dotProduct
            self._accuracy_2 = tf.reduce_sum(p,1)
            self._accuracy_3 = tf.reduce_sum(z_,1)


        learning_rate = 1.0e-2
        with tf.name_scope('train'):
            # Optimizer

            train_op = tf.train.AdamOptimizer(learning_rate).minimize(self._cross_entropy);
            self._train_op=train_op

    def train(self):

        # training samples
        medley = MedleyDBAnalyser(rootDir=rootDir, destDir=destDir)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            NUM_EPOCHS = 1000
            BATCH_SIZE=5
            for epoch in range(0,NUM_EPOCHS):
                i=0
                batches = medley.getSampleBatch(batchSize=BATCH_SIZE)
                for batch in batches:
                    if i == 0:
                        start_time = time.time()
                    if i % 10 == 0:
                        train_accuracy_1 = self._accuracy_1.eval(feed_dict={
                                self.x: batch[0], self.z_: batch[1]})
                        train_accuracy_2 = self._accuracy_2.eval(feed_dict={
                                self.x: batch[0], self.z_: batch[1]})
                        train_accuracy_3 = self._accuracy_3.eval(feed_dict={
                                self.x: batch[0], self.z_: batch[1]})

                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print('epoch# ',epoch,'\t iteration# ',i,'\tsamples#',i*BATCH_SIZE, '\ttime[s]=', elapsed_time)
                        print('\t\t dotProduct=',train_accuracy_1)
                        print('\t\t sum of predictions=',train_accuracy_2)
                        print('\t\t sum of lables=',train_accuracy_3 )
                        start_time = time.time()
                        time.sleep(0.25)

                    self._train_op.run(feed_dict={self.x: batch[0], self.z_: batch[1]})
                    i += 1
                print('end of epoch...',epoch)
            print('end of training...')



def main(_):
    v = VocalSeparator()
    v.buildModel()
    v.train()



if __name__ == '__main__':

    import sys
    print(sys.version)
    tensorboard_log_dir = '/path/to/tensorboard/log'
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')

    parser.add_argument(
        '--log_dir',
        type=str,
        default=tensorboard_log_dir,
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


