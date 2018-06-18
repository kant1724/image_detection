from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tensorflow as tf

ip_addr = open('./train_mode', encoding="utf8").readlines()[1].replace('\n', '')

app = Flask(__name__, static_url_path="/static") 
CORS(app)


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
@app.route('/image_to_text', methods=['POST'])
def image_to_text():
    img_file = request.data
    f = io.BytesIO(img_file)
    image = Image.open(f)
    dim = len(image.mode)
    (im_width, im_height) = image.size
    
    image_np = np.array(image.getdata()).reshape((im_height, im_width, dim)).astype(np.uint8)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    (image_tensor, boxes, scores, classes, num_detections) = sess.run(
        [image_tensor, boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    
    cls = np.squeeze(classes)[0 : 20]
    score = np.squeeze(scores)[0 : 20]
    left = [bb[1] for bb in np.squeeze(boxes)[0 : 20]]
    for i in range(len(left)):
        idx = np.argsort(left)
        cls = np.array(cls)[idx]
        score = np.array(score)[idx]
        left = np.array(left)[idx]
    
    cc = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    
    a = []
    b = []
    c = []
    
    for i in range(len(score)):
        if score[i] > 0.98:
            a.append(cls[i])
            b.append(score[i])
            c.append(left[i])
    result = ""
    for i in range(len(c)):
        if len(result) == 6:
            break
        if i + 1 < len(c) - 1 and c[i + 1] - c[i]  <= 0.07:
            continue
        result += cc[int(a[i]) - 1]
    
    res = {'text' : result}
    return jsonify(res)

if (__name__ == "__main__"): 
    app.run(host=ip_addr, port = 5005) 
