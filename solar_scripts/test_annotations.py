import sys
import cv2

source = sys.argv[1]
image_path = 'data/images/training/'
labels_path = 'data/labels/training/'
filename = f'{image_path}{source}_test.jpg'

print(f"Reading: {image_path}{source}.jpg")
image = cv2.imread(f"{image_path}{source}.jpg")

class_list = ['sunspot']
colors = [(0, 0, 255)]

height, width, _ = image.shape

with open(f"{labels_path}{source}.txt", "r") as file1:
    for line in file1.readlines():
        split = line.split(" ")

        # getting the class id
        class_id = int(split[0])
        color = colors[class_id]
        clazz = class_list[class_id]

        # getting the xywh bounding box coordinates
        x, y, w, h = float(split[1]), float(split[2]), float(split[3]), float(split[4])

        # re-scaling xywh to the image size
        box = [int((x - 0.5*w)* width), int((y - 0.5*h) * height), int(w*width), int(h*height)]
        cv2.rectangle(image, box, color, 1)
        # cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        # cv2.putText(image, class_list[class_id], (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

cv2.imwrite(filename, image)

print(f"Exported: {filename}")