import os
import sys
import argparse
import pandas as pd
from numpy import *
from PIL import Image

def main(args):
    sunspots_df = read_csv(args.csv)
    image = open_image(args.image)

    for index, row in sunspots_df.iterrows():
        x_centroid, y_centroid = find_centroid(row, image)
        sunspots_df.at[index, 'x_centroid'] = x_centroid
        sunspots_df.at[index, 'y_centroid'] = y_centroid
    
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
    df.sort_values('confidence')
    print(df)
    return df

def open_image(image_path):
    print(f"Reading {image_path}")
    image = Image.open(image_path)
    return image

def find_centroid(sunspot, image):
    x = sunspot["x"]
    y = sunspot["y"]
    w = sunspot["width"]
    h = sunspot["height"]
    image_width, image_height = image.size

    print(f"Processing sunspot {x}, {y}")

    # Crop image
    box = (x, y, x+w, y+h)
    im = image.crop(box)

    # List single values from RGBA pixels
    pixels = list(im.getdata())
    data = []
    for i in range(len(pixels)):
        data.append(pixels[i][0])

    # Determine vertices around sunspot
    vertices = [
        (0,0),
        (w,0),
        (w,h),
        (0,h)
    ]

    # Count vertices and denote number by n
    n = len(vertices)

    # Add x & y values from vertices and divide by sum of n
    sum_x, sum_y = [ sum(row[i] for row in vertices) for i in range(len(vertices[0])) ]
    x_centroid = sum_x / n
    y_centroid = sum_y / n

    return x + x_centroid, y + y_centroid

def save_output(sunspots_df, output_path):
    print(f"Saving {output_path}")
    print(sunspots_df)
    sunspots_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
