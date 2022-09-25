import os
import sys
import argparse
import pandas as pd
import numpy as np
import cv2
import re

def main(args):
    sunspots_df = read_csv(args.csv)
    image = open_image(args.image)
    vertices = []
    centroids = []

    for index, row in sunspots_df.iterrows():
        x_centroid, y_centroid, min_brightness, max_brightness, verts = find_centroid(row, image, args.output)
        sunspots_df.at[index, 'x_centroid'] = x_centroid
        sunspots_df.at[index, 'y_centroid'] = y_centroid
        sunspots_df.at[index, 'min_brightness'] = min_brightness
        sunspots_df.at[index, 'max_brightness'] = max_brightness

        # For image marking
        for i in range(len(verts)):
            verts[i] = (verts[i][0] + sunspots_df.at[index, 'x'], verts[i][1] + sunspots_df.at[index, 'y'])
        vertices.extend(verts)
        centroids.append((int(x_centroid), int(y_centroid)))

    print(centroids)

    # Save output files
    save_image(vertices, centroids, image, args.output)
    save_output(sunspots_df, args.output)

def save_image(vertices, centroids, image, output_path):
    # Draw centroid on cropped image as a red pixel
    for i in range(len(centroids)):
        image[centroids[i][1]][centroids[i][0]] = [0, 0, 255]

    # Draw vertices on cropped image as blue pixels
    for i in range(len(vertices)):
        image[vertices[i][1]][vertices[i][0]] = [255, 0, 0]

    # Save image
    print(f"Saving {output_path.rsplit('.', 1)[0]}.centroids.png")
    cv2.imwrite(f"{output_path.rsplit('.', 1)[0]}.centroids.png", image)

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
    #image = Image.open(image_path)
    image = cv2.imread(image_path)
    return image

def find_centroid(sunspot, image, output_path):
    threshold = 0.4
    min_edges = 2
    offset_x = int(sunspot['x'])
    offset_y = int(sunspot['y'])
    w = int(sunspot["width"])
    h = int(sunspot["height"])
    image_width, image_height, channels = image.shape

    # Crop image
    im = image[offset_y:offset_y+h, offset_x:offset_x+w]

    # List single values from RGBA pixels
    pixel_data = []
    for x in range(0, im.shape[0]):
        for y in range(0, im.shape[1]):
            pixel_data.append(im[x, y][0])

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
    for i in range(len(data)):
        if data[i] < threshold:
            data[i] = 0

    # Reshape to 2D array
    arr = np.reshape(data, (-1, w))

    # Find Contours
    contour_arr = np.copy(arr)
    for y in range(h):
        for x in range(w):
            if (arr[y, x] > 0):
                 # Add up elements under threshold as edges
                edges = 0
                if x+1 > w-1 or arr[y, x+1] == 0:
                    edges += 1
                if x-1 < 0 or arr[y, x-1] == 0:
                    edges += 1
                if y+1 > h-1 or arr[y+1, x] == 0:
                    edges += 1
                if y-1 < 0 or arr[y-1, x] == 0:
                    edges += 1

                # Select element as contour if it has enough edges
                if edges >= min_edges:
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
    centroid_x = sum_x / n
    centroid_y = sum_y / n

    # Print data
    np.set_printoptions(precision=2, linewidth=200)
    print(f"Sunspot {offset_x}, {offset_y}")
    print(arr)
    print(vertices)
    for i in range(len(vertices)):
        print(vertices[i][0] + offset_x, vertices[i][1] + offset_y)
    print(contour_arr)

    return offset_x + centroid_x, offset_y + centroid_y, min_value, max_value, vertices

def save_output(sunspots_df, output_path):
    print(f"Saving {output_path}")
    print(sunspots_df)
    sunspots_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
