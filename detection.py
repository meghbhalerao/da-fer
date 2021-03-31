from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import sys
from utils.utils import *
import matplotlib.pyplot as plt
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display
import os

mtcnn = MTCNN(keep_all=True)
# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

images_path  = "/home/megh/projects/fer/da-fer/data/aisin-datasets/ID01_face/Natural/"

img = Image.open("/home/megh/projects/fer/da-fer/data/random-images/bbt.jpeg")
img = Image.open(os.path.join(images_path,os.listdir(images_path)[0]))
img = img.convert('RGB')
total_images = len(os.listdir(images_path))
can_detect = 0
cant_detect_paths = []
for idx, image in enumerate(os.listdir(images_path)):
    img_path = os.path.join(images_path,image)
    img = Image.open(os.path.join(images_path,image)).convert('RGB')
    # Get cropped and prewhitened image tensor
    print("Image Path is: ", img_path)
    print(img.size)
    boxes, probablities, landmarks_points = mtcnn.detect(img,landmarks=True)
    # Visualize
    print(landmarks_points,boxes)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img)
    ax.axis('off')
    if landmarks_points is not None:
        for box, landmark in zip(boxes, landmarks_points):
            #ax.plot(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
            #ax.plot(*np.meshgrid(box[[1, 2]], box[[0, 3]]))
            #ax.plot(*np.meshgrid(box[[0,1,2,3]]))
            ax.scatter(landmark[:, 0], landmark[:, 1], s=35)
        can_detect = can_detect + 1
        fig.show()
        plt.savefig("./output_images/pic_%s.png"%(str(image)))

    elif landmarks_points is None:
        cant_detect_paths.append(image)
        plt.savefig("./output_images/pic_%s_undetected.png"%(str(image)))
  
    print(boxes)
print("Percentage of Images Detected: ", can_detect/total_images * 100)
#plt.savefig("pic.png")



"""
for image in os.listdir(images_path):
    img = Image.open(os.path.join(images_path,image)) 
    frame_draw = img.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

    for landmark in landmarks_points:
        print(landmark)
        draw.point(landmark,fill="black")
    frame_draw.show()

    d = display.display(draw, display_id=True)
"""

sys.exit()

# For a model pretrained on VGGFace2
model = InceptionResnetV1(pretrained='vggface2').eval()

# For a model pretrained on CASIA-Webface
model = InceptionResnetV1(pretrained='casia-webface').eval()

# For an untrained model with 100 classes
model = InceptionResnetV1(num_classes=100).eval()

# For an untrained 1001-class classifier
model = InceptionResnetV1(classify=True, num_classes=1001).eval()
# If required, create a face detection pipeline using MTCNN:
#mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)


# Draw faces
frame_draw = img.copy()
draw = ImageDraw.Draw(frame_draw)
for box in boxes:
    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    draw.point()
frame_draw.show()

d = display.display(draw, display_id=True)
