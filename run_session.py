import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
np.set_printoptions(threshold=np.nan)
# Necessary to explicitly add the models/ folders

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

if __name__ == '__main__':

    print('Running __main__')
    # Importing object_detection modules from TensorFlow. The compiler might say that
    # there is an error in these lines, but the truth is that when sys.path.append lines
    # are executed, then the compiler is able to import these two modules
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util

    print('Model preparation...')
    '''
    Any model exported using the export_inference_graph.py tool can be loaded here simply by 
    changing PATH_TO_CKPT to point to a new .pb file. By default we use an "SSD with Mobilenet" 
    model here. See the detection model zoo for a list of other models that can be run 
    out-of-the-box with varying speeds and accuracies.
    '''

    MODEL_NAME = './training_data/4. pb_files'
    F_NAME = '/frozen_inference_graph.pb'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + F_NAME

    # List of the strings that is used to add correct label for each box.
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
    '''
    Label maps map indices to category names, so that when our convolution network 
    predicts 5, we know that this corresponds to airplane. Here we use internal utility 
    functions, but anything that returns a dictionary mapping integers to appropriate 
    string labels would be fine
    '''
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print('Label map loaded')

    print('Detection task started...')
    PATH_TO_TEST_IMAGES_DIR = 'test_data/images'
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 't_{}.jpg'.format(i)) for i in range(0, 10) ]
    
    IMAGE_SIZE = (24, 16)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=tf.ConfigProto(intra_op_parallelism_threads=20)) as sess:
            with tf.device('/cpu:0'):
                for image_path in TEST_IMAGE_PATHS:
                    image = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (image_tensor, boxes, scores, classes, num_detections) = sess.run(
                        [image_tensor, boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    #print(np.squeeze(classes).astype(np.int32))
                    #print(np.squeeze(scores))
                    #print(np.squeeze(boxes))
                    #classes, boxes = sess.run([classes, boxes], feed_dict={image_tensor: image_np_expanded})
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
                    print(a)
                    print(c)
                    res = ""
                    for i in range(len(c)):
                        if len(res) == 5:
                            break
                        if i + 1 < len(c) - 1 and c[i + 1] - c[i]  <= 0.07:
                            continue
                        res += cc[int(a[i]) - 1]
                    
                    print(res)
                    '''                
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=1,
                        min_score_thresh=0.95)
                    plt.figure(figsize=IMAGE_SIZE)
                    plt.imshow(image_np)
                    plt.show()
                    '''