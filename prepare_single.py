import os

from holisticDetector import HolisticDetector
import random
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def load_a_frozen_model(path_to_ckpt):
    """

    :param path_to_ckpt: string
                         checkpoint file which contains the graph information to be loaded
    :return: detection_graph : tf.Graph() object
                             : the graph information from ckpt files is loaded into this tf.Graph() object
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


loadedModel = load_a_frozen_model('./frozen_inference_graph.pb')


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width):
    """Transforms the box masks back to full image masks.

    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.

    Args:
      box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
      boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
             corners. Row i contains [ymin, xmin, ymax, xmax] of the box
             corresponding to mask i. Note that the box corners are in
             normalized coordinates.
      image_height: Image height. The output mask will have the same height as
                    the image height.
      image_width: Image width. The output mask will have the same width as the
                   image width.

    Returns:
      A tf.float32 tensor of size [num_masks, image_height, image_width].
    """


holisticDetector = HolisticDetector(min_detection_confidence=0.7)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def detect_person(image):

    output_dict = run_inference_for_single_image(
        image, loadedModel)
    boxes = output_dict['detection_boxes']
    rectangle_pts = boxes[0, :] * np.array(
        [image.shape[0], image.shape[1], image.shape[0], image.shape[1]])

    IMAGE_SIZE = (12, 8)

    # plt.figure(figsize=IMAGE_SIZE)

    # plt.imshow(image[int(rectangle_pts[0]): int(rectangle_pts[2]),
    #                  int(rectangle_pts[1]): int(rectangle_pts[3])])
    # plt.savefig("temp.png")

    return rectangle_pts


def process_image(image):

    print('start::::::::::::')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    (thresh, black) = cv2.threshold(image, 255, 255, cv2.THRESH_BINARY)
    rect_pts = detect_person(image)
    print('---------------------- PERSON DETECTED -----------------------')

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    results = holisticDetector.findLandMarks(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    holisticDetector.drawLeftLandMarks(black, results)

    black = cv2.flip(black, 1)
    black = black[int(rect_pts[0]): int(rect_pts[2]),
                  int(rect_pts[1]): int(rect_pts[3])]

    reshaped_img = cv2.resize(black, (500, 500))

    # image = cv2.flip(image, 1)
    # image = image[int(rect_pts[0]): int(rect_pts[2]),
    #               int(rect_pts[1]): int(rect_pts[3])]

    # reshaped_img_1 = cv2.resize(image, (500, 500))

    return reshaped_img


def preprocessVideo(file, outputPath):
    try:

        raw_clip = VideoFileClip(file)

        bg_clip = raw_clip.fl_image(process_image)

        bg_clip.write_videofile(outputPath, audio=False)

    except Exception as e:
        print('--------- error occured ------------')
        print(e)
