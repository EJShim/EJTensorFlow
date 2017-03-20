import random
from PyQt5.QtWidgets import QApplication
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class E_Manager:
    def __init__(self, window):

        self.window = window
        self.sess = tf.Session()
        self.plotX = []
        self.plotY = []

        self.InitMNistData()
        self.InitNetwork()
        self.InitFunctions()




    def InitMNistData(self):
        # Import MNIST data
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets("./MNistData", one_hot=True)

    def GetRandomImage(self):
        #Get Random Image
        testImageSet = self.mnist.test.images
        randidx = random.randrange(0, testImageSet.shape[0])

        result = np.reshape(testImageSet[randidx], (28, 28))

        return [result, randidx]

    def InitNetwork(self):
        print("init network")

        # # Parameter
        self.learning_rate = 0.001
        self.training_iters = 50000
        self.batch_size = 128
        self.display_step = 10

        # Network Parameters
        self.n_input = 784 # MNIST data input (img shape: 28*28)
        self.n_classes = 10 # MNIST total classes (0-9 digits)
        self.dropout = 0.75 # Dropout, probability to keep units

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        # self.visual = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)



        #Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.n_classes]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }


    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y
    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def avgpool2d(self, x, k=2):
        return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


    # Create model
    def conv_net(self, x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])


        # Max Pooling (down-sampling)
        maxpool1 = self.maxpool2d(conv1, k=2)


        # Convolution Layer
        conv2 = self.conv2d(maxpool1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        maxpool2 = self.avgpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        dense = tf.reshape(maxpool2, [-1, weights['wd1'].get_shape().as_list()[0]])

        fc1 = tf.add(tf.matmul(dense, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        # fc1 = tf.nn.dropout(fc1, dropout)


        # Output, softmax prediction
        result = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

        out = {'input':x, 'conv1':conv1, 'maxpool1':maxpool1, 'conv2':conv2, 'maxpool2':maxpool2, 'result':result}

        return out


    def InitFunctions(self):
        # Construct model
        self.network = self.conv_net(self.x, self.weights, self.biases, self.keep_prob)
        self.pred = self.network['result']

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    def RunTrainning(self):

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        self.sess.run(init)

        step = 1
        # Keep training until reach max iterations
        # QApplication.processEvents()
        while step * self.batch_size < self.training_iters:
            batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)
            # Run optimization op (backprop)
            self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.dropout})

            if step % self.display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = self.sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.})
                # print(self.weights['wc1'].eval()[0][0][0][0])

                print("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                self.plotX.append(step*self.batch_size)
                self.plotY.append(loss)

                QApplication.processEvents()
                self.window.DrawGraph()

            step += 1

        self.window.SetLog("Optimization Finished!")


    def SaveModel(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "./models/model.ckpt")

        self.window.SetLog("Model Saved in ./models/model.ckpt")

    def LoadModel(self):
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('./models/model.ckpt.meta')
        saver.restore(self.sess, "./models/model.ckpt")

        self.window.SetLog("Model Loaded from ./models/model.ckpt")


    def RunPrediction(self, image):

        if(self.sess):
            image = np.reshape(image, (1, 784))
            network = self.sess.run(self.network, feed_dict={self.x: image, self.keep_prob:1.} )


            inp = network['input']
            plot = self.window.m_figure.add_subplot(2,3,1)
            plot.set_title("Input Image")
            plot.imshow(inp[0,:,:,0], cmap=plt.get_cmap('gray'))
            plot.axis('off')

            #Visualize First Conv
            conv1 = network['conv1']
            plot = self.window.m_figure.add_subplot(2,3,2)
            plot.set_title("First Layer(32)")
            plot.imshow(conv1[0,:,:,0], cmap=plt.get_cmap('gray'))
            plot.axis('off')

            #Visualize Second Conv
            conv2 = network['conv2']
            plot = self.window.m_figure.add_subplot(2,3,3)
            plot.set_title("Second Layer(64)")
            plot.imshow(conv2[0,:,:,0], cmap=plt.get_cmap('gray'))
            plot.axis('off')

            #prediction
            pred = network['result']
            pred = np.multiply(pred, 0.0001)
            pred = self.softmax(pred)
            pred = np.multiply(pred, 100.0)
            res = np.argmax(pred[0])

            cweight = self.sess.run(self.weights['out'])



            #Class Activation Map
            predweights = cweight[:, res:res+1]
            camsum = np.zeros((14, 14))
            for j in range(64): #Number of conv2 output filters
                camsum = camsum + predweights[j]*conv2[0, :, :, j]

            #Average of Camsum
            camavg = camsum / 256
            plot = self.window.m_figure.add_subplot(2,3,4)
            plot.set_title("Activation Map")
            plot.imshow(camavg)
            plot.axis('off')



            log = "Predicted Digit : " + str(res)
            self.window.SetLog(log)
        else:
            print("not trained")



    # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
