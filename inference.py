import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib as plt
import cv2
import os

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def extract_face_for_test(test_dir, test_output_dir):
    PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
    PATH_TO_LABELS = './protos/face_label_map.pbtxt'
    NUM_CLASSES = 2
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    with detection_graph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(graph=detection_graph, config=config) as sess:
        counter = 1
        if not os.path.exists(test_output_dir):
            os.mkdir(test_output_dir)
        print('extracting face from test images....')
        for image_name in os.listdir(test_dir):
            if image_name != '.DS_Store':
                image = cv2.imread(test_dir + image_name)
                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                faces = []
                faces = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4)
                im_width = image_np.shape[1]
                im_height = image_np.shape[0]
                for (ymin, xmin, ymax, xmax) in faces:
                    (x, y, w, h) = (int(xmin*im_width), int(ymin*im_height), int(xmax*im_width) - int(xmin*im_width), int(ymax*im_height) - int(ymin*im_height))
                    roi_cropped = image[y:y+h, x:x+w]
                    cv2.imwrite(test_output_dir + str(counter) + '.jpg', roi_cropped)
                    counter += 1

def extract_face():
    PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
    PATH_TO_LABELS = './protos/face_label_map.pbtxt'
    NUM_CLASSES = 2
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    with detection_graph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(graph=detection_graph, config=config) as sess:
        counter = 1
        faces_list = []
        max_width = 0
        max_height = 0
        raw_image_path = sys.argv[1]
        combined_image_path = sys.argv[2]
        if not os.path.exists(combined_image_path):
            os.mkdir(combined_image_path)
        print('extracting face from images....')
        for image_name in os.listdir(raw_image_path):
            if image_name != '.DS_Store':
                image = cv2.imread(raw_image_path + image_name)
                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
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
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                #print(boxes.shape, boxes)
                #print(scores.shape,scores)
                #print(classes.shape,classes)
                #print(num_detections)
                # Visualization of the results of a detection.

                faces = []
                faces = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4)
                im_width = image_np.shape[1]
                im_height = image_np.shape[0]


                for (ymin, xmin, ymax, xmax) in faces:
                    (x, y, w, h) = (int(xmin*im_width), int(ymin*im_height), int(xmax*im_width) - int(xmin*im_width), int(ymax*im_height) - int(ymin*im_height))
                    roi_cropped = image[y:y+h, x:x+w]
                    if w > max_width:
                        max_width = w
                    if h > max_height:
                        max_height = h
                    faces_list.append(roi_cropped)
                    if len(faces_list) == 50:
                        blank_image = np.zeros((max_height * 5, max_width * 10, 3), np.uint8)
                        for j in range(5):
                            for k in range(10):
                                im_width = faces_list[(j*10) + k].shape[1]
                                im_height = faces_list[(j*10) + k].shape[0]
                                complement = np.zeros((max_height, max_width, 3), np.uint8)
                                complement[0:im_height, 0:im_width] = faces_list[(j*10) + k]
                                blank_image[max_height*j:max_height*(j+1), max_width*k:max_width*(k+1)] = complement
                        faces_list = []
                        max_width = 0
                        max_height = 0
                        cv2.imwrite(combined_image_path + str(counter) + '.jpg', blank_image)
                        counter += 1
        
                    

        
