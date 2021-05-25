
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

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()
