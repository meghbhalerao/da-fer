from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import sys

from numpy.lib.type_check import imag
from utils.utils import *
import matplotlib.pyplot as plt
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display
import os
import imutils
from imutils import face_utils
import dlib

def main():
    method = 'mtcnn' # options: dlib, mtcnn
    print("Method to detect faces being used: %s"%(method))
    if method == 'mtcnn':
        mtcnn = MTCNN(keep_all=True)
    elif method == 'dlib':
        weight_path = os.path.join("./model_weights/dlib_weights.dat")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor()


    main_path  = "/home/megh/projects/fer/da-fer/data/aisin-datasets/face-images/"
    images_types = ['_leftfront_','_rightfront_','_right45_','_front_','_rightside_', '_rightside_', '_left45_']
    detected_images_number = np.zeros(len(images_types))
    total_images_number = np.zeros(len(images_types))

    total_images = 0
    cant_detect_paths = []
    can_detect  = 0

    for person_id in os.listdir(main_path):
        person_path = os.path.join(main_path, person_id)
        for emotion_folder in os.listdir(person_path):
            emotion_path = os.path.join(person_path,emotion_folder)
            for idx, image_name in enumerate(os.listdir(emotion_path)):
                img_path = os.path.join(emotion_path,image_name)
                if method == 'mtcnn': # using PIL for image ops in mtcnn mode
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    print(img.size)
                elif method == 'dlib': # using opencv for image ops in dlib mode
                    img = cv2.imread(img_path)
                    img = imutils.resize(img, width=500)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Get cropped and prewhitened image tensor
                print("Image Name is: ", os.path.basename(img_path))
                if method == "mtcnn":
                    do_mtcnn_detect(mtcnn,img,can_detect,cant_detect_paths,image_name,images_types,detected_images_number,total_images_number)
                elif method == "dlib":
                    do_dlib_detect(detector,predictor,image_name,img)
                total_images = total_images + 1
                print(can_detect/total_images)
                
    print("Percentage of Images Detected: ", 1 - len(cant_detect_paths)/total_images * 100)
    print(detected_images_number)
    print(total_images_number)
    print("Images detected by individual class are:  ", detected_images_number/total_images_number)
if __name__ == "__main__":
    main()
"""
img = Image.open("/home/megh/projects/fer/da-fer/data/random-images/blur.jpg")
img = Image.open(os.path.join(images_path,os.listdir(images_path)[0]))
img = img.convert('RGB')

for idx, image_name in enumerate(os.listdir(images_path)):
    img_path = os.path.join(images_path,image_name)
    if method == 'mtcnn': # using PIL for image ops in mtcnn mode
        img = Image.open(os.path.join(images_path,image_name))
        img = img.convert('RGB')
        print(img.size)

    elif method == 'dlib': # using opencv for image ops in dlib mode
        img = cv2.imread(img_path)
        img = imutils.resize(img, width=500)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get cropped and prewhitened image tensor
    print("Image Path is: ", img_path)
    
    if method == "mtcnn":
        do_mtcnn_detect(mtcnn,img,can_detect,cant_detect_paths,image_name)
    elif method == "dlib":
        do_dlib_detect(detector,predictor,image_name,img)
    
    print("Percentage of Images Detected: ", can_detect/total_images * 100)
#plt.savefig("pic.png")

"""
