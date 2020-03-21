from flask import Flask, request, make_response, abort
from flask_restplus import Api, Resource
from PIL import Image
import tensorflow as tf
import requests
import time
import tempfile
from io import BytesIO 
import os
import json
import cv2

import sys
sys.path.append("./keras-yolo3")
from slim_yolo import YOLO

app = Flask(__name__)
api = Api(app)
app.config.from_pyfile('app.cfg')

if os.getenv('FLASK_CONFIG_FILE') is not None:
    app.config.from_envvar('FLASK_CONFIG_FILE')
# TODO Can be changed with environment variables
yolo = YOLO(
    model_path= 'keras-yolo3/model_data/yolo.h5',
    anchors_path= 'keras-yolo3/model_data/yolo_anchors.txt',
    classes_path= 'keras-yolo3/model_data/coco_classes.txt'
)
graph = tf.get_default_graph()
timeout_seconds = float(app.config['API_TIMEOUT_SECONDS'])

@api.route('/health')
class PingResource(Resource):
    @api.doc(responses={200: 'OK'})
    def get(self):
        return  {"status": "ok"}

@api.route('/image/capture')
class ImageCaptureResource(Resource):
    device_num=0
    delay=1
    window_name="frame"

    @api.doc(responses={200: 'OK'})
    def get(self):
        cap = cv2.VideoCapture(self.device_num)
        if not cap.isOpened():
            return  {"status": "ng"}

        ret, frame = cap.read()
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(self.delay) & 0xFF
        tmp_data = tempfile.mkstemp(prefix="captured", suffix=".jpg")
        cv2.imwrite(tmp_data[1], frame)

        cv2.destroyWindow(self.window_name)
        return  {
            "status": "ok",
            "captured_image": tmp_data[1]
        }


@api.route('/image/detect')
class ImageDetectResource(Resource):
    @api.doc(responses={200: 'OK'})
    # for test
    def get(self):
        image_path = request.args.get('path')
        with graph.as_default():
            image = Image.open(image_path)
            image_data, annotations = yolo.detect_object(image, rectangle_class='person')
            tmp_file_name = _write_temp_data(image_data, "annotated", ".jpg")
            results = {
                "local_file_name": tmp_file_name,
                "annotations": annotations
            }
            return results

def _write_temp_data(data, prefix, suffix):
    tmp_data = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    with open(tmp_data[1],'wb') as f:
        f.write(data)

    return tmp_data[1]    


if __name__ == '__main__':
    app.run(host='0.0.0.0') 
