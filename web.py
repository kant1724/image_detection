from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import json
import numpy as np
import io
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_CKPT = './training_data/4. pb_files/capchar_500_images_250000_steps.pb'

PATH_TO_LABELS = os.path.join('training_data/1. image_and_label_data', 'label_map.pbtxt')
NUM_CLASSES = 10

print('Loading a (frozen) Tensorflow model into memory...')
detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('Model loaded')

print('Loading label map...')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print('Label map loaded')
print('Detection task started...')
PATH_TO_TEST_IMAGES_DIR = 'test_images'

sess = tf.Session(graph=detection_graph)

class S(BaseHTTPRequestHandler): 
    def _set_headers(self): 
        self.send_response(200) 
        self.send_header('Content-type', 'text/html') 
        self.end_headers() 
 
    def do_GET(self): 
        self._set_headers() 
        self.wfile.write(b'{{"result" : "123456"}}')
 
    def do_POST(self): 
        self._set_headers() 
        length = int(self.headers.get('content-length'))
        buffer = self.rfile.read(length)
        output = io.BytesIO(buffer)
        '''
        output2 = io.BytesIO()
        lines = output.readlines()
        for i, line in enumerate(lines):
            if i < 3 or i == len(lines) - 1:
                continue
            output2.write(line)
        '''
        img = Image.open(output)
        num = self.process(img)
        result = '{"result" : "' + num + '"}'
        self.wfile.write(result.encode())
    
    def process(self, image):
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (image_tensor, boxes, scores, classes, num_detections) = sess.run(
            [image_tensor, boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        b = np.squeeze(boxes)[0 : 6]
        r = []
        for bb in b:
            r.append(bb[1])
        index = []
        for i in range(6):
            ii = np.argmin(r)
            r[ii] = 999999 + i
            index.append(ii)
        
        cc = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        result1 = ""
        for i in index:
            result1 += cc[np.squeeze(classes)[i].astype(np.int32) - 1]
            
        return result1
    
def run(server_class=HTTPServer, port=5005): 
    server_address = ('', port) 
    httpd = HTTPServer(server_address, S)
    print('Starting httpd...') 
    httpd.serve_forever()     

run()
