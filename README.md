## Solar YOLOv5

YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com/yolov5">Ultralytics</a>
open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

This fork includes scripts to help track sunspots.

## 📘Installation Instructions
1. Clone the solar-yolo repository.

```
git clone https://github.com/ChrisToumanian/solar-yolo
```

2. You will need Python >=3.8 and PIP. Check using `python -V`.

3. Create and enter a Python environment for the project.

```
cd solar-yolo
python -m venv venv
. venv/bin/activate
```

If you’re using PowerShell in Windows, use this command to activate the environment:

```
.\venv\Scripts\activate
```

4. Install the required Python modules contained in requirements.txt.

```
pip install -r requirements.txt
```

> :warning: **If you have an Nvidia GeForce 30 series GPU**: You need to install a special version of PyTorch that's cu116 compatible. Download the wheel for TorchVision from http://download.pytorch.org/whl/torchvision/

```
pip install --extra-index-url https://download.pytorch.org/whl/cu116/ "torch==1.12.1+cu116"
pip install torchvision-0.13.1+cu116-cp310-cp310-linux_x86_64.whl
```

## 🧪Test Installation
Inside the data directory there is an image, `data/images/bus.jpg`. Use this to test YOLOv5.

1. Use `detect.py` to detect the objects within the image using a pre-trained model capable of detecting common objects, yolov5s. It’s pretty fun to try with any other image as well.

```
python detect.py --weights yolov5s.pt --source data/images/bus.jpg --save-txt --save-conf --save-crop
```

2. If this was successful it will save the results in `yolov5/runs/exp/`. Each run produces a new directory, `exp`, `exp2`, `exp3`, etc. The following files should appear in the following directories:

`runs/exp/bus.jpg` shows the original image plus bounding boxes around four people and one bus, along with labels and confidences.

`runs/exp/labels/bus.txt` contains data showing labels, bounding boxes, and confidences for each detected feature.

`runs/exp/crops/` contains cropped images of each feature.

## 🌠Set up FITS image directory

Set up a directory and script to convert Flexible Image Transport System (FITS) files to images.

1. Get an example FITS solar intensity file from the Solar Dynamics Observatory, e.g. `hmi.im_45s.20160920_000000_TAI.2.continuum.fits`.

2. Create the directory and move example FITS file into it.

```
mkdir data/fits_images
mkdir data/fits_images/20160920
mv ~/hmi.im_45s.20160920_000000_TAI.2.continuum.fits data/fits_images/20160920
```

3. Install Astropy Python module.

```
pip install astropy
```

4. Convert the fits file to a PNG image using `solar/save_fits_image.py`. This will save a PNG image with the same name as the FITS file in the same directory.

```
python save_fits_image.py data/fits_images/20160920/hmi.im_45s.20160920_000000_TAI.2.continuum.fits
```

## 🗒️Annotate images in preparation for training

Small 224 x 224 pixel images clipped from a variety of FITS solar intensitygram images are used.

1. Use an image editing tool such as Photoshop or GIMP to clip out a few sunspots and determine the bounding box of each sunspot in pixel coordinates. Name them in the format `0001.jpg`.

![8834c83c-54d8-44a4-8b76-eaf0d8bc350f](https://user-images.githubusercontent.com/4646154/191718553-0f927540-8257-4b9d-8c73-5fd5b37be277.jpg)

2. YOLOv5 requires an annotation file named in the format `0001.txt` that contains a label and relative floating-point coordinates relating to each image used for training the model. The first field is the label. Because there  is only one label in this case, for sunspots, it is `0`. The following fields represent the sunspot’s center-x, center-y, width, height:

```
0 0.4819524727039178 0.4996959965746093 0.2810190537358167 0.3126696638835367
0 0.46996360522372077 0.46708627702847355 0.27814172554056943 0.3088332262898737
```

Relative coordinates are used because they do not require knowledge of the image’s width or height. For example, the rectangle’s abscissa is calculated as a floating-point percentage of the image’s overall width, offset by half the width of the rectangle to obtain the center x-coordinate.

```
def bbox_to_yolo_bbox(x, y, w, h, image_w, image_h):
    x1 = (x + w/2) / image_w
    y1 = (y + h/2) / image_h
    x2 = w / image_w
    y2 = h / image_h
    return x1, y1, x2, y2
```

3. Use the attached script `convert_to_yolo_coordinates.py` to convert the pixel bounding box XYWH to relative floats.

```
# convert_to_yolo_coordinates.py <image width> <image height> <x> <y> <w> <h>
python convert_to_yolo_coordinates.py 224 224 75 70 59 64
> 0.46651785714285715 0.45535714285714285 0.26339285714285715 0.2857142857142857
```

## 🏋️Train a new model for sunspot detection

The training images must be saved within `data/images/training/` and `data/images/validation/`. The training label text files must be saved within `data/labels/training/` and `data/labels/validation/`. Ideally, 10% of the images and their corresponding labels will be located in the validation directory with a large dataset, but with a small dataset you may place all images and labels in validation as well.

1. Use the `dataset.yaml` file to define the classes, training and validation directories.

```
# labels: /data/labels/training/
# labels_val: /data/labels/validation/
train: data/images/training/
val: data/images/validation/

# number of classes
nc: 1

# class names
names: ['sunspot']
```

2. Run this command to train the model using `train.py` and wait for it to complete. It may take some time if you’re using the large weights. This should produce a model and corresponding metadata in the directory `runs/train/exp/`. Check the results saved there.

```
python train.py --img 224 --data dataset.yaml --weights yolov5l.pt
```

## 🔭Run sunspot feature detection

You can run feature detection on any FITS PNG image by running this command, using `detect.py`. This optionally saves crops of each sunspot and saves them to the `runs/detect/exp/` along with the output text file.

```
python detect.py --weights runs/train/exp/weights/best.pt --img 4090 --conf 0.1 --source data/fits_images/20160920/hmi.im_45s.20160920_000000_TAI.2.continuum.fits.png --save-txt --save-conf --save-crop --hide-labels --line-thickness 1
```

## ☀️Convert output to delivery format

The script `convert_coordinates.py` is attached below and used to convert the output representing each sunspot. Here is an example.

```
label,confidence,x_center_relative,y_center_relative,width_relative,height_relative,x_center,y_center,width,height,x_1,y_1,x_2,y_2
0,0.101128,0.33728,0.495117,0.00463867,0.00341797,1379,2025,19,14,1370,2018,1388,2032
0,0.104789,0.0675049,0.433472,0.00561523,0.0100098,276,1773,23,41,265,1753,287,1793
0,0.106513,0.346436,0.490356,0.00292969,0.00317383,1417,2006,12,13,1411,2000,1423,2012
```

Run the `convert_coordinates.py` script to convert YOLOv5’s output to our own format:

```
python convert_coordinates.py -f runs/detect/exp/labels/hmi.im_45s.20160920_000000_TAI.2.continuum.fits.txt -o runs/detect/exp/labels/hmi.im_45s.20160920_000000_TAI.2.continuum.fits.csv -w 4090 -h 4090 -v -H -d ","
```

Run this to add centroids and save the final file to be delivered:

```
python3 centroid.py -c runs/detect/exp/labels/hmi.im_45s.20160920_000000_TAI.2.continuum.fits.csv -i data/fits_images/20160920/hmi.im_45s.20160920_000000_TAI.2.continuum.fits.png -o runs/detect/exp/labels/hmi.im_45s.20160920_000000_TAI.2.continuum.fits_centroids.csv
```
