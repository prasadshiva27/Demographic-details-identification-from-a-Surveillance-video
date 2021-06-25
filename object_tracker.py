import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import math
from collections import Counter
from collections import deque
import keras

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


# Return true if line segments AB and CD intersect
# @staticmethod
def intersect(A, B, C, D):
    # return Camera.ccw(A, C, D) != Camera.ccw(B, C, D) and Camera.ccw(A, B, C) != Camera.ccw(A, B, D)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# @staticmethod
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# @staticmethod
def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))

#Function to predict Image demographics
def getImageDetails(model, img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  image_data_flat = img.shape[0]*img.shape[1]
  if image_data_flat > 120*60:
      img = cv2.resize(img,(60,120),cv2.INTER_AREA)
  else:
      img = cv2.resize(img,(60,120),cv2.INTER_LINEAR)

  # plt.imshow(img)
  img = np.expand_dims(img, axis=0)

  classes =['personalFemale', 'personalMale', 'personalLess15', 'personalLess30',
  'personalLess45' ,'personalLess60', 'personalLarger60']
  shortForms = ['F','M','(0-15)','(16-30)','(31-45)','(46-60)','(>60)']

  proba = []
  proba = model.predict([img])  #Get probabilities for each class
  # proba[0] = [i for i in proba[0] if i > 0.4]
  sorted_categories = []
  sorted_categories = np.argsort(proba[0])[:-6:-1]  #Get class names for top 5 categories
  if shortForms[sorted_categories[0]] == 'M' or shortForms[sorted_categories[0]] == 'F':
    return (shortForms[sorted_categories[0]], shortForms[sorted_categories[1]])
  else:
    return (shortForms[sorted_categories[1]], shortForms[sorted_categories[0]])
  # return [shortForms[sorted_categories[0]], shortForms[sorted_categories[1]]]


def main(_argv):

    # Initialising the Keras model to predict Demographics of image
    model = keras.models.load_model('model_data/New_32CL_5LR_43Epoc')

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0

    
    memory = {}
    total_counter = 0
    up_count = 0
    down_count = 0

    class_counter = Counter()  # store counts of each detected class
    already_counted = deque(maxlen=50)  # temporary memory for storing counted IDs
    intersect_info = []  # initialise intersection list
    cropped_images = []
    demographic_details = []

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        # Masking the mirror in the video
        #pts = np.array([[0,123], [114,103], [124, 362], [15, 463], [0, 327]],np.int32)
        #pts = pts.reshape((-1, 1, 2))
        #cv2.fillPoly(frame, [pts],(255,255,255))

        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # Addition of Line in the middle of frame
        line = [(0, int(0.55 * frame.shape[0])), (int(frame.shape[1]), int(0.55 * frame.shape[0]))]
        # cv2.line(frame, line[0], line[1], (0, 255, 255), 2)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # Tracking midpoints
            midpoint = track.tlbr_midpoint(bbox)
            origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])  # get midpoint respective to botton-left

            if track.track_id not in memory:
                    memory[track.track_id] = deque(maxlen=2)

            memory[track.track_id].append(midpoint)
            previous_midpoint = memory[track.track_id][0]

            origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])

            cv2.line(frame, midpoint, previous_midpoint, (0, 255, 0), 2)

             # Add to counter and get intersection details
            if intersect(midpoint, previous_midpoint, line[0], line[1]) and track.track_id not in already_counted:
                class_counter[class_name] += 1
                total_counter += 1

                # draw red line
                # cv2.line(frame, line[0], line[1], (0, 0, 255), 2)

                already_counted.append(track.track_id)  # Set already counted for ID to true.

                # intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
                angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                # intersect_info.append([class_name, origin_midpoint, angle, intersection_time])

                if angle > 0:
                    up_count += 1
                if angle < 0:
                    down_count += 1
                    
                    # cropping image
                    xmin, ymin, xmax, ymax = bbox
                    cropped_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                    cropped_images.append(cropped_img)

                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    image_data_flat = cropped_img.shape[0]*cropped_img.shape[1]
                    if image_data_flat > 120*60:
                        cropped_img = cv2.resize(cropped_img,(60,120),cv2.INTER_AREA)
                    else:
                        cropped_img = cv2.resize(cropped_img,(60,120),cv2.INTER_LINEAR)
                    cv2.imwrite('./images/'+str(track.track_id)+'.jpg',cropped_img)
                    k = getImageDetails(model, cropped_img)
                    demographic_details.append(k)
                    print(k)
                    # croppedImages.append(cropped_img)

            
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)  # WHITE BOX
            # cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
            #                 1.5e-2 * frame.shape[0], (0, 255, 0), 4)

            # Delete memory of old tracks.
            # This needs to be larger than the number of tracked objects in the frame.
            if len(memory) > 50:
                del memory[list(memory)[0]]

            if len(cropped_images)>10:
                cropped_images.pop(0)
                demographic_details.pop(0)


            # Draw total count.
            cv2.putText(frame, "Total: {} ({} Out, {} In)".format(str(total_counter), str(up_count),
                        str(down_count)), (int(0.05 * frame.shape[1]), int(0.1 * frame.shape[0])), 0,
                        1.5e-3 * frame.shape[0], (0, 255, 255), 2)

            # cv2.putText(frame, "Total: {} ({} up, {} down)".format(str(total_counter), str(up_count),
            #             str(down_count)), (int(0.05 * frame.shape[1]), int(0.1 * frame.shape[0])), 0,
            #             1.5e-3 * frame.shape[0], (0, 255, 255), 8)


            # Paste the cropped image on to the frame
            if cropped_images:
              crop_start_w, crop_start_h = 5, int(frame.shape[0]) - int(2 * 0.1 * frame.shape[0])
              
              for i in range(len(cropped_images)):
                  resized_image = cv2.resize(cropped_images[i], (int(0.1 * frame.shape[0]),int(2*0.1 * frame.shape[0])))
                  cropped_h, cropped_w = resized_image.shape[:2]
                  frame[crop_start_h : crop_start_h + cropped_h, crop_start_w + (cropped_w*i) : crop_start_w + (cropped_w*i) + cropped_w] = resized_image
                  cv2.putText(frame, demographic_details[i][0] + '_' + demographic_details[i][1], 
                                        (crop_start_w + (cropped_w*i),crop_start_h - 5 ),cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5e-3 * frame.shape[0],(40,40,172),thickness = 1 )
            
            
        # draw bbox on screen
            # color = colors[int(track.track_id) % len(colors)]
            # color = [i * 255 for i in color]
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            # cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
