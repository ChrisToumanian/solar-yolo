import sys

def yolo_bbox_to_bbox(x, y, w, h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2

def bbox_to_yolo_bbox(x, y, w, h, image_w, image_h):
    x1 = (x + w/2) / image_w
    y1 = (y + h/2) / image_h
    x2 = w / image_w
    y2 = h / image_h
    return x1, y1, x2, y2

def main():
    image_w = float(sys.argv[1])
    image_h = float(sys.argv[2])
    x = float(sys.argv[3])
    y = float(sys.argv[4])
    w = float(sys.argv[5])
    h = float(sys.argv[6])
    data = bbox_to_yolo_bbox(x, y, w, h, image_w, image_h)
    print(*data, sep=' ')

main()
