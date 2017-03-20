import tensorflow as tf
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib

class NodeLookup(object):

  def __init__(self, model_dir):
    label_lookup_path = os.path.join(model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    uid_lookup_path = os.path.join(model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)


    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string


    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]


    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]




class E_InceptionV3:
    def __init__(self, window):


        self.DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        self.model_dir = "./imageNet"
        self.image_file = ""
        self.num_top_predictions = 5;

        self.window = window
        self.sess = None

        self.Initialize()

    def Initialize(self):
        #Download Inception V3 Graph
        self.LoadInceptionV3()

        #Create Graph
        self.CreateGraph();

        #Predict Image

    def LoadInceptionV3(self):
        #Download and Extract model tar file
        # dest_directory = self.FLAGS.model_dir
        dest_directory = self.model_dir

        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)

        filename = self.DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)

        print("filepath : ", filepath)

        if not os.path.exists(filepath):
            print("not exist")
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>>Downloading %s %.1f%%' % (filename, float(count*block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(self.DATA_URL, filepath, _progress);
            print()
            statinfo = os.stat(filepath)
            print("succesfully downloaded", filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def CreateGraph(self):

        with tf.gfile.FastGFile(os.path.join(self.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _=tf.import_graph_def(graph_def, name='')


    def PredictImage(self, imagePath):
        if not tf.gfile.Exists(imagePath):
            tf.loggin.fatal("File Does not Exist %s", imagePath)

        image_data = tf.gfile.FastGFile(imagePath, 'rb').read()



        with tf.Session() as self.sess:

            #decode Image Data
            image = self.sess.run(tf.image.decode_jpeg(image_data))
            plot = self.window.m_figure.add_subplot(111)
            plot.set_title("Input Image (ImgNet trained inception v3)")
            plot.imshow(image)
            plot.axis('off')

            softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')
            predictions = self.sess.run(softmax_tensor, {'DecodeJpeg/contents:0':image_data})
            predictions = np.squeeze(predictions) #what is this?

            node_lookup = NodeLookup(self.model_dir)

            top_k = predictions.argsort()[-self.num_top_predictions:][::-1]

            log = ""
            for node_id in top_k:
                human_string = node_lookup.id_to_string(node_id)
                score = predictions[node_id]

                #Log
                log += "%s (score = %.5f) \n" % (human_string, score)


            #Add Log
            self.window.SetLog(log, True)
