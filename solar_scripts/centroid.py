import os
import sys
import argparse
import pandas as pd
import numpy as np
from PIL import Image

def main(args):
    sunspots_df = read_csv(args.csv)
    image = open_image(args.image)

    for index, row in sunspots_df.iterrows():
        x_centroid, y_centroid, min_brightness, max_brightness = find_centroid(row, image)
        sunspots_df.at[index, 'x_centroid'] = x_centroid
        sunspots_df.at[index, 'y_centroid'] = y_centroid
        sunspots_df.at[index, 'min_brightness'] = min_brightness
        sunspots_df.at[index, 'max_brightness'] = max_brightness
    
    save_output(sunspots_df, args.output)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", help="Input CSV file", type=str, required=True)
    parser.add_argument("-i", "--image", help="Input image file", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output CSV file", type=str, required=True)
    args = parser.parse_args()
    return args

def read_csv(csv_path):
    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[
        'label',
        'x_center_relative',
        'y_center_relative',
        'width_relative',
        'height_relative',
        'x_2',
        'y_2'
    ])
    df = df.rename(columns={"x_1": "x", "y_1": "y"})
    df["x_centroid"] = 0
    df["y_centroid"] = 0
    df["min_brightness"] = 0
    df["max_brightness"] = 0
    df.sort_values('confidence')
    return df

def open_image(image_path):
    print(f"Reading {image_path}")
    image = Image.open(image_path)
    return image

def find_centroid(sunspot, image):
    offset_x = sunspot['x']
    offset_y = sunspot['y']
    w = int(sunspot["width"])
    h = int(sunspot["height"])
    image_width, image_height = image.size

    # Crop image
    box = (offset_x, offset_y, offset_x+w, offset_y+h)
    im = image.crop(box)

    # List single values from RGBA pixels
    pixels = list(im.getdata())
    pixel_data = []
    for i in range(len(pixels)):
        pixel_data.append(pixels[i][0])

    # Min-max normalize
    data = pixel_data
    min_value = min(data)
    max_value = max(data)

    for i in range(len(data)):
        data[i] = (data[i] - min_value) / (max_value - min_value)

    # Invert array
    for i in range(len(data)):
        data[i] = 1 - data[i]

    # Threshold cut-off
    threshold = 0.65
    for i in range(len(data)):
        if data[i] < threshold:
            data[i] = 0

    # Reshape to 2D array
    arr = np.reshape(data, (-1, w))

    # Find Contours
    contour_arr = np.copy(arr)
    for y in range(h):
        for x in range(w):
            if (arr[y, x] > 0
                and (
                    (x+1 > w-1 or arr[y, x+1] == 0)
                    or (x-1 < 0 or arr[y, x-1] == 0)
                    or (y+1 > h-1 or arr[y+1, x] == 0)
                    or (y-1 < 0 or arr[y-1, x] == 0)
                )
            ):
                contour_arr[y, x] = 1
            else:
                contour_arr[y, x] = 0

    # Create vertices from contour
    vertices = []
    for y in range(h):
        for x in range(w):
            if contour_arr[y, x] == 1:
                vertices.append((x, y))

    # # Count vertices and denote number by n
    n = len(vertices)

    # # Add x & y values from vertices and divide by sum of n
    sum_x, sum_y = [ sum(row[i] for row in vertices) for i in range(len(vertices[0])) ]
    x_centroid = sum_x / n
    y_centroid = sum_y / n

    return offset_x + x_centroid, offset_y + y_centroid, min_value, max_value

def save_output(sunspots_df, output_path):
    print(f"Saving {output_path}")
    print(sunspots_df)
    sunspots_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
