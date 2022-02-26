import argparse

import cv2
import numpy as np
from time import time
import tflite_runtime.interpreter as tflite
from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer, CvSource, VideoMode
from networktables import NetworkTablesInstance
import cv2
import collections
import json
import sys


class ConfigParser:
    def __init__(self, config_path):
        self.team = -1

        # parse file
        try:
            with open(config_path, "rt", encoding="utf-8") as f:
                j = json.load(f)
        except OSError as err:
            print("could not open '{}': {}".format(config_path, err), file=sys.stderr)

        # top level must be an object
        if not isinstance(j, dict):
            self.parseError("must be JSON object", config_path)

        # team number
        try:
            self.team = j["team"]
        except KeyError:
            self.parseError("could not read team number", config_path)

        # cameras
        try:
            self.cameras = j["cameras"]
        except KeyError:
            self.parseError("could not read cameras", config_path)

    def parseError(self, str, config_file):
        """Report parse error."""
        print("config error in '" + config_file + "': " + str, file=sys.stderr)


class PBTXTParser:
    def __init__(self, path):
        self.path = path
        self.file = None

    def parse(self):
        with open(self.path, 'r') as f:
            self.file = ''.join([i.replace('item', '') for i in f.readlines()])
            blocks = []
            obj = ""
            for i in self.file:
                if i == '}':
                    obj += i
                    blocks.append(obj)
                    obj = ""
                else:
                    obj += i
            self.file = blocks
            label_map = []
            for obj in self.file:
                obj = [i for i in obj.split('\n') if i]
                name = obj[2].split()[1][1:-1]
                label_map.append(name)
            self.file = label_map

    def get_labels(self):
        return self.file


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

    def scale(self, sx, sy):
        """Returns scaled bounding box."""
        return BBox(xmin=sx * self.xmin,
                    ymin=sy * self.ymin,
                    xmax=sx * self.xmax,
                    ymax=sy * self.ymax)


class Tester:
    def __init__(self, config_parser):
        print("Initializing TFLite runtime interpreter")
        try:
            model_path = "model.tflite"
            self.interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            self.hardware_type = "Coral Edge TPU"
        except:
            print("Failed to create Interpreter with Coral, switching to unoptimized")
            model_path = "unoptimized.tflite"
            self.interpreter = tflite.Interpreter(model_path)
            self.hardware_type = "Unoptimized"

        self.interpreter.allocate_tensors()

        print("Getting labels")
        parser = PBTXTParser("map.pbtxt")
        parser.parse()
        self.labels = parser.get_labels()

        print("Connecting to Network Tables")
        ntinst = NetworkTablesInstance.getDefault()
        ntinst.startClientTeam(config_parser.team)
        ntinst.startDSClient()
        self.entry = ntinst.getTable("ML").getEntry("detections")

        self.coral_entry = ntinst.getTable("ML").getEntry("coral")
        self.fps_entry = ntinst.getTable("ML").getEntry("fps")
        self.resolution_entry = ntinst.getTable("ML").getEntry("resolution")
        self.temp_entry = []

        print("Starting camera server")
        
        cs = CameraServer.getInstance()
        arg = 0
        name = "USB Camera %d" % arg
        camera = UsbCamera(name, arg)

        mpeg_server = cs.startAutomaticCapture(camera=camera)
        mpeg_server.setCompression(30)
        camera_config = config_parser.cameras[0]
        WIDTH, HEIGHT = camera_config["width"], camera_config["height"]
        camera.setResolution(WIDTH, HEIGHT)
        self.cvSink = cs.getVideo() 
        
        self.img = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)

        self.output = CvSource('Axon', VideoMode.PixelFormat.kMJPEG, WIDTH, HEIGHT, 30)
        axonMjpegServer = cs.startAutomaticCapture(camera=self.output)
        axonMjpegServer.setCompression(30)

        self.frames = 0

        self.coral_entry.setString(self.hardware_type)
        self.resolution_entry.setString(str(WIDTH) + ", " + str(HEIGHT))

    def run(self):
        print("Starting mainloop")
              
        lower_blue = np.array([90,50,70])
        upper_blue = np.array([128,255,255])
        
        # lower boundary RED color range values; Hue (0 - 10)
        lower_red1 = np.array([0, 100, 20])
        upper_red1 = np.array([10, 255, 255])
 
        # upper boundary RED color range values; Hue (160 - 180)
        lower_red2 = np.array([160,100,20])
        upper_red2 = np.array([179,255,255])
    
        while True:
            start = time()
            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame_cv2 = self.cvSink.grabFrame(self.img)
            if not ret:
                print("Image failed")
                continue
            
            # input
            scale = self.set_input(frame_cv2)

            # Run inference
            self.interpreter.invoke()

            # output
            boxes, class_ids, scores, x_scale, y_scale = self.get_output(scale)
            for i in range(len(boxes)):
                if scores[i] >= .5:

                    class_id = class_ids[i]
                    if np.isnan(class_id):
                        continue

                    class_id = int(class_id)
                    if class_id not in range(len(self.labels)):
                        continue
                        
                    # get 'region of interest' image from bounding box                   
                    ymin, xmin, ymax, xmax = boxes[i]
                        
                    bbox = BBox(xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax).scale(x_scale, y_scale)
                        
                    # check bbox validity
                    height, width, channels = frame_cv2.shape
                    if (0 <= ymin < ymax <= height) and (0 <= xmin < xmax <= width):
                                   
                        """           
                        Check to see if the bounding box is "mostly" square. Cargo farther away or farther
                        off angle from the center of the field of view seem to lose their squareness
                        """                                  
                        aspect_ratio_threshold = 0.30
                        
                        aspect_ratio = xmax / ymax                       
                                           
                        #if (aspect_ratio <= (1.0 + aspect_ratio_threshold)) and (aspect_ratio >= (1.0 - aspect_ratio_threshold)):
                        
                        ymin, xmin, ymax, xmax = int(bbox.ymin), int(bbox.xmin), int(bbox.ymax), int(bbox.xmax)
                                                                 
                        roi = frame_cv2[ymin:ymax, xmin:xmax]
                        
                        # Convert ROI image to HSV format
                        hsv_roi_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                        # Looks like there's a boatload of ways to create red and blue masks which seems stupid
                        lower_red_mask = cv2.inRange(hsv_roi_image, lower_red1, upper_red1)
                        upper_red_mask = cv2.inRange(hsv_roi_image, lower_red2, upper_red2)
                        red_mask = lower_red_mask + upper_red_mask
                       
                        total_pixels = bbox.xmax * bbox.ymax
                        
                        """
                        Based on experiments, it appears as though any ball with more than the following percentages
                        of red or blue pixels can be classified as a red or blue cargo respectively. This assumes
                        that there isn't other red/blue objects on the field that look close enough to a cargo ball.
                        """
                        red_threshold_percentage = 0.5 # yes, this is 0.5% ... just slightly over zero
                        blue_threshold_percentage = 0.5 # yes, this is 0.5% ... just slightly over zero
                        
                        count_red = cv2.countNonZero(red_mask)               
                        
                        # The 'scores' value is actually the confidence that it is a "Ball" not a Red or Blue Cargo
                        if ((count_red / total_pixels) * 100.0) > red_threshold_percentage:

                            #self.label_frame(frame_cv2, "C:" + str(count_red) + " P:" + str(total_pixels), boxes[i], scores[i], x_scale, y_scale)
                            #self.label_frame(frame_cv2, "A:" + str(aspect_ratio), boxes[i], scores[i], x_scale, y_scale)
                            self.label_frame(frame_cv2, "RedCargo", boxes[i], scores[i], x_scale, y_scale)
                        else: 
                            blue_mask = cv2.inRange(hsv_roi_image, lower_blue, upper_blue)
                        
                            count_blue = cv2.countNonZero(blue_mask)
                            
                            if ((count_blue / total_pixels) * 100.0) > blue_threshold_percentage:
                                #self.label_frame(frame_cv2, "C:" + str(count_blue) + " P:" + str(total_pixels), boxes[i], scores[i], x_scale, y_scale)
                                #self.label_frame(frame_cv2, "A:" + str(aspect_ratio), boxes[i], scores[i], x_scale, y_scale)
                                self.label_frame(frame_cv2, "BlueCargo", boxes[i], scores[i], x_scale, y_scale)
                            #else:                    
                                #self.label_frame(frame_cv2, "Unknown", boxes[i], scores[i], x_scale, y_scale)
                                #self.label_frame(frame_cv2, "R:" + str(count_red) + "B:" + str(count_blue) + "P:" + str(total_pixels), boxes[i], scores[i], x_scale, y_scale)

            self.output.putFrame(frame_cv2)
            self.entry.setString(json.dumps(self.temp_entry))
            
            self.temp_entry = []
            
            if self.frames % 100 == 0:
                print("Completed", self.frames, "frames. FPS:", (1 / (time() - start)))
            if self.frames % 10 == 0:
                self.fps_entry.setNumber((1 / (time() - start)))
            self.frames += 1

    def label_frame(self, frame, object_name, box, score, x_scale, y_scale):
        ymin, xmin, ymax, xmax = box
        score = float(score)
        bbox = BBox(xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax).scale(x_scale, y_scale)

        height, width, channels = frame.shape
        # check bbox validity
        if not 0 <= ymin < ymax <= height or not 0 <= xmin < xmax <= width:
            return frame

        ymin, xmin, ymax, xmax = int(bbox.ymin), int(bbox.xmin), int(bbox.ymax), int(bbox.xmax)
        self.temp_entry.append({"label": object_name, "box": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax},
                                "confidence": score})
 
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

        # Draw label
        # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, score * 100)  # Example: 'person: 72%'
        label_size, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)  # Get font size
        label_ymin = max(ymin, label_size[1] + 10)  # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base - 10),
                      (255, 255, 255), cv2.FILLED)
        # Draw label text
        cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        return frame

    def input_size(self):
        """Returns input image size as (width, height) tuple."""
        _, height, width, _ = self.interpreter.get_input_details()[0]['shape']
        return width, height

    def set_input(self, frame):
        """Copies a resized and properly zero-padded image to the input tensor.
        Args:
          frame: image
        Returns:
          Actual resize ratio, which should be passed to `get_output` function.
        """
        width, height = self.input_size()
        h, w, _ = frame.shape
        new_img = np.reshape(cv2.resize(frame, (300, 300)), (1, 300, 300, 3))
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], np.copy(new_img))
        return width / w, height / h

    def output_tensor(self, i):
        """Returns output tensor view."""
        tensor = self.interpreter.get_tensor(self.interpreter.get_output_details()[i]['index'])
        return np.squeeze(tensor)

    def get_output(self, scale):
        boxes = self.output_tensor(0)
        class_ids = self.output_tensor(1)
        scores = self.output_tensor(2)

        width, height = self.input_size()
        image_scale_x, image_scale_y = scale
        x_scale, y_scale = width / image_scale_x, height / image_scale_y
        return boxes, class_ids, scores, x_scale, y_scale


if __name__ == '__main__':
    config_file = "/boot/frc.json"
    config_parser = ConfigParser(config_file)
    tester = Tester(config_parser)
    tester.run()
