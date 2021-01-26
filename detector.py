import numpy as np
import cv2
import tensorflow as tf

class Detector:
    def __init__(self, model_path):
        # loading the model and building the detection function
        print('Loading Model...')
        self.model_path = model_path
        self.detection_fn = tf.saved_model.load(self.model_path)        
        print('Done Loading...')   
        
        # self.detection_graph = tf.Graph()
#         Loading the frozen model
        # with self.detection_graph.as_default():
            # frozen_graph = tf.compat.v1.GraphDef()
            # print("***** save model path - {} *****", self.model_path)
            # with tf.io.gfile.GFile(self.model_path, 'rb') as f:                
                # frozen_graph.ParseFromString(f.read())
                # tf.import_graph_def(frozen_graph, name='')
# 
        # self.default_graph = self.detection_graph.as_default()
        # self.sess = tf.Session(graph=self.detection_graph)
# 
        # self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        # self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        # self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):                
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        detections = self.detection_fn(input_tensor)
        return detections


        # image_np_expanded = np.expand_dims(image, axis=0)
        # 
#        Detection starts to run
        # start_time = time.time()
        # (boxes, scores, classes, num) = self.sess.run(
            # [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            # feed_dict={self.image_tensor: image_np_expanded})
        # end_time = time.time()
# 
        # im_height, im_width, _ = image.shape
        # boxes_list = [None for i in range(boxes.shape[1])]
        # for i in range(boxes.shape[1]):
            # boxes_list[i] = (
                # int(boxes[0, i, 0] * im_height),
                # int(boxes[0, i, 1] * im_width),
                # int(boxes[0, i, 2] * im_height),
                # int(boxes[0, i, 3] * im_width))
# 
        # return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])
# 
    def close(self):
        return




if __name__ == "__main__":
    model_path = '/Users/janmajayamall/Desktop/fyp_detector/models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model'
    print("***** Loading Model *****")
    odapi = Detector(model_path)
    print("***** Model Loaded *****")
    threshold = 0.7

    cap = cv2.VideoCapture('/Users/janmajayamall/Desktop/fyp_detector/videoplayback.mp4')

    # seats = [[(975, 610), (1381, 1079)], [(530, 610), (983, 1079)], [(618, 382), (987, 651)], [(957, 382), (1383, 651)]]

    while True:
        success, img = cap.read()
        if not success:
            break

        # img = cv2.resize(img, (1280, 720))

        # Crop the whole frame down to one of the seats
        # (x0, y0), (x1, y1) = seats[1][0], seats[1][1]
        # img = img[y0:y1, x0:x1]

        # boxes, scores, classes, num = odapi.processFrame(img)

        detections = odapi.processFrame(img)        
        # Visualization of the results of a detection.
        # for i, seat in enumerate(seats):
        #     (x0, y0), (x1, y1) = seat[0], seat[1]
        #     cv2.putText(img, "seat{}".format(i), (x0+10, y0+10), cv2.FONT_HERSHEY_PLAIN, 0.9, RED)
        #     cv2.rectangle(img, (x0, y0), (x1, y1), RED, 2)

        # for i in range(len(boxes)):
            # box = boxes[i]
          #  checking human detected
            # if classes[i] == 1 and scores[i] > threshold:  
                # cv2.putText(
                    # img, "human: {}".format(scores[i]),
                    # (box[1]+10, box[0]+10), cv2.FONT_HERSHEY_PLAIN,
                    # 0.9, BLUE)
                # cv2.rectangle(
                    # img,
                    # (box[1], box[0]),
                    # (box[3], box[2]),
                    # BLUE, 2)
           # checking chair detected 
            # elif classes[i] == 62 and scores[i] > threshold:  
                # cv2.putText(
                    # img, "chair: {}".format(scores[i]),
                    # (box[1]+10, box[0]+10), cv2.FONT_HERSHEY_PLAIN,
                    # 0.9, GREEN)
                # cv2.rectangle(
                    # img,
                    # (box[1], box[0]),
                    # (box[3], box[2]),
                    # GREEN, 2)
            # elif scores[i] > threshold:
                # cv2.putText(
                    # img, "obj{}: {}".format(classes[i], scores[i]),
                    # (box[1]+10, box[0]+10),
                    # cv2.FONT_HERSHEY_PLAIN,
                    # 0.9, RED)
                # cv2.rectangle(
                    # img,
                    # (box[1], box[0]),
                    # (box[3], box[2]),
                    # RED, 2)
# 
        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    cap.release()

    cv2.destroyAllWindows()