import PIL
from PIL.Image import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from imutils import face_utils

def detect_faces(image,model):
    face_coordinates, prob = model.detect(image)
    return face_coordinates, prob

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

# this is for a single image
def do_mtcnn_detect(mtcnn,img,can_detect,cant_detect_paths,image_name,image_types,detected_images_number,total_images_number):
    boxes, probablities, landmarks_points = mtcnn.detect(img,landmarks=True)
    #print(landmarks_points,boxes,image_name)
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
        print(can_detect)
        #fig.show()
        plt.savefig("./output_images/pic_%s.png"%(str(image_name)))

    elif landmarks_points is None:
        if cant_detect_paths is not None:
            cant_detect_paths.append(image_name)
        plt.savefig("./output_images/pic_%s_undetected.png"%(str(image_name)))
    
    print(cant_detect_paths)
    # incrementing the specific class of image counter accoring to the image path
    if landmarks_points is not None:
        for idx, image_type in enumerate(image_types):
            if str(image_type) in image_name:
                detected_images_number[idx] = detected_images_number[idx] + 1
    
    for idx, image_type in enumerate(image_types):
        if str(image_type) in image_name:
            total_images_number[idx] = total_images_number[idx] + 1


def do_dlib_detect(detector,predictor,image_name,img):
    rects = detector(img, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        
        for (x, y) in shape:
            print("something")
            cv2.circle(img, (x, y), 1, (250, 0, 0), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    pass