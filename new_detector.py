import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import time
from collections import namedtuple
import firebase_admin
from firebase_admin import credentials, firestore

CvColor = namedtuple('CvColor', 'b g r')
BLUE = CvColor(255, 0, 0)
GREEN = CvColor(0, 255, 0)
RED = CvColor(0, 0, 255)

# model zoo object detection tensorflow v1
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
# also a good guide - https://towardsdatascience.com/object-detection-by-tensorflow-1-x-5a8cb72c1c4b

class ObjectDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.detection_graph = tf.Graph()

        # Load the frozen model
        with self.detection_graph.as_default():
            frozen_graph = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as f:
                frozen_graph.ParseFromString(f.read())
                tf.import_graph_def(frozen_graph, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Define input and output tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        # print("Elapsed Time:", end_time-start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (
                int(boxes[0, i, 0] * im_height),
                int(boxes[0, i, 1] * im_width),
                int(boxes[0, i, 2] * im_height),
                int(boxes[0, i, 3] * im_width))
        
        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        # self.default_graph.close()


if __name__ == "__main__":

    # connect with firebase
    cred = credentials.Certificate("./empty-seat-detection-firebase-adminsdk-73j10-0c3f9f253c.json")
    firebase_admin.initialize_app(cred)
    firestore_db = firestore.client()

    firestore_db.collection(u'places').document(u'Ay2AW8ckRaO3GKbaqZXP').set({'occupiedSeats': 0, 'totalSeats': 2})        

    model_path = '/Users/janmajayamall/Desktop/fyp_detector/models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
    odapi = ObjectDetector(model_path)
    threshold = 0.7

    cap = cv2.VideoCapture('/Users/janmajayamall/Desktop/fyp_detector/data/video/IMG_0251.MOV')

    seats = [(258, 1081, 827, 1501), (274, 379, 924, 817)]
    occupancy_count = 0

    while True:
        success, img = cap.read()
        # img = cv2.imread("/Users/janmajayamall/Desktop/fyp_detector/data/images/1.jpg", 1)
        if not success:
            break

        current_frame_count = 0 

        for i in range(len(seats)):
            seat = seats[i]
            new_image = img[seat[0]-200:seat[2], seat[1]:seat[3]]
            new_boxes, new_scores, new_classes, new_num = odapi.processFrame(new_image)            
            count_flag = True
            for j in range(len(new_classes)):
                    if new_classes[j] == 1 and new_scores[j] > threshold:
                        if (count_flag == True):
                            current_frame_count += 1
                            count_flag = False
                            break
            
        if current_frame_count != occupancy_count:
            occupancy_count = current_frame_count
            print(occupancy_count)
            # send request
            # firestore_db.collection(u'places').document(u'Ay2AW8ckRaO3GKbaqZXP').set({'occupiedSeats': occupancy_count, 'totalSeats': 2})        
            

        # boxes, scores, classes, num = odapi.processFrame(img)

        # for i in range(len(boxes)):
        #     box = boxes[i]
        #     if classes[i] == 62 and scores[i] > threshold:                
                
        #         score = scores[i]                

        #         new_img = img[box[0]-200:box[2], box[1]:box[3]]

        #         cv2.imshow("preview", new_img)
        #         print("score ", scores[i])
        #         key = cv2.waitKey(1)
        #         if key & 0xFF == ord('q'):
        #            break
        #         print("human inference")
        #         new_boxes, new_scores, new_classes, new_num = odapi.processFrame(new_img)
        #         print("human inference done")
        #         count_flag = True

        #         for j in range(len(new_classes)):
        #             if new_classes[j] == 1 and scores[j] > threshold:
        #                 if (count_flag == True):
        #                     current_frame_count += 1
        #                     count_flag = False
        


     
        #     if classes[i] == 1 and scores[i] > threshold:  # Human            
        #         cv2.putText(
        #             img, "human: {}".format(scores[i]),
        #             (box[1]+10, box[0]+10), cv2.FONT_HERSHEY_PLAIN,
        #             0.9, BLUE)
        #         cv2.rectangle(
        #             img,
        #             (box[1], box[0]),
        #             (box[3], box[2]),
        #             BLUE, 2)
        #     elif classes[i] == 62 and scores[i] > threshold:  # Chair
        #         print(box, scores[i])
        #         cv2.putText(
        #             img, "chair: {}".format(scores[i]),
        #             (box[1]+10, box[0]+10), cv2.FONT_HERSHEY_PLAIN,
        #             0.9, GREEN)
        #         cv2.rectangle(
        #             img,
        #             (box[1], box[0]),
        #             (box[3], box[2]),
        #             GREEN, 2)
        #     elif scores[i] > threshold:
        #         cv2.putText(
        #             img, "obj{}: {}".format(classes[i], scores[i]),
        #             (box[1]+10, box[0]+10),
        #             cv2.FONT_HERSHEY_PLAIN,
        #             0.9, RED)
        #         cv2.rectangle(
        #             img,
        #             (box[1], box[0]),
        #             (box[3], box[2]),
        #             RED, 2)

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    
    firestore_db.collection(u'places').document(u'Ay2AW8ckRaO3GKbaqZXP').set({'occupiedSeats': 0, 'totalSeats': 2})        
    
    # cap.release()
    cv2.destroyAllWindows()