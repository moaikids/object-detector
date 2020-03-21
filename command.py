import sys
sys.path.append("./keras-yolo3")
from slim_yolo import YOLO
from io import BytesIO

from PIL import Image

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            print(img)
            image = Image.open(img)
        except Exception as e:
            print(e)
            continue

        try:
            image_data, annotations = yolo.detect_object(image, rectangle_class='person')
            print(annotations)
            Image.open(BytesIO(image_data)).show()
        except Exception as e:
            print(e)
            continue



if __name__ == '__main__':
    yolo = YOLO(
        model_path= 'keras-yolo3/model_data/yolo.h5',
        anchors_path= 'keras-yolo3/model_data/yolo_anchors.txt',
        classes_path= 'keras-yolo3/model_data/coco_classes.txt'
    ) 
    detect_img(yolo)
