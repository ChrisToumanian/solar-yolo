# ==========================================================================================
# Calculate sunspot metrics
# Maintainers: Christopher Toumanian, cct_580@usc.edu
#              Jimmy Wen, jswen@usc.edu
# ==========================================================================================
import os
import sys
import argparse
import pandas as pd
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import scipy.signal
from astropy.io import fits

# ==========================================================================================
# Main & Arguments
# ==========================================================================================
def main(args):
    sunspots_df = read_csv(args.csv)
    image = open_fits_image(args.image, args.fits_header)
    vertices = []
    centroids = []

    for index, row in sunspots_df.iterrows():
        area, average_intensity, x_centroid, y_centroid, min_intensity, max_intensity, centroid_intensity, verts = get_sunspot_metrics(row, image, args.output, args.threshold, args.min_conv_sum, args.max_conv_sum)
        sunspots_df.at[index, 'x_centroid'] = x_centroid
        sunspots_df.at[index, 'y_centroid'] = y_centroid
        sunspots_df.at[index, 'area'] = area
        sunspots_df.at[index, 'centroid_intensity'] = centroid_intensity
        sunspots_df.at[index, 'average_intensity'] = average_intensity
        sunspots_df.at[index, 'min_intensity'] = min_intensity
        sunspots_df.at[index, 'max_intensity'] = max_intensity

        # For image marking
        for i in range(len(verts)):
            verts[i] = (verts[i][0] + sunspots_df.at[index, 'x'], verts[i][1] + sunspots_df.at[index, 'y'])
        vertices.extend(verts)
        centroids.append((int(x_centroid), int(y_centroid)))

    # Save output files
    if args.output_centroid_image:
        save_image(vertices, centroids, image, args.output)
    save_output(sunspots_df, args.output)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", help="Input CSV file", type=str, required=True)
    parser.add_argument("-i", "--image", help="Input fits image file", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output CSV file", type=str, required=True)
    parser.add_argument("-v", "--output_centroid_image", help="Output image of vertices and centroids", action='store_true')
    parser.add_argument("-t", "--threshold", help="Threshold between sunspot and photosphere", type=float, required=False, default=0.562)
    parser.add_argument("-n", "--min_conv_sum", help="Minimum convolution sum to set vertices", type=int, required=False, default=3)
    parser.add_argument("-m", "--max_conv_sum", help="Maximum convolution sum to set vertices", type=int, required=False, default=5)
    parser.add_argument("-f", "--fits_header", help="Header location of image data in fits file", type=int, required=False, default=0)
    parser.add_argument("-s", "--sort_by", help="Sort output by a specified parameter", type=str, required=False, default="area")
    parser.add_argument("-a", "--ascending", help="Sort ascending", action='store_true')
    args = parser.parse_args()
    return args

# ==========================================================================================
# I/O
# ==========================================================================================
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
    df.sort_values('confidence')
    return df

def open_fits_image(image_path, image_data_header_location):
    print(f"Reading {image_path}")
    image_file = open(image_path, "rb")
    hdu_list = fits.open(image_file)
    hdu_list.info()
    image_data = hdu_list[image_data_header_location].data
    return image_data
    
def save_output(sunspots_df, output_path):
    # Sort by specified parameter
    sunspots_df = sunspots_df.sort_values(args.sort_by, ascending=args.ascending)
    print(f"Saving {output_path}")
    print(sunspots_df)
    sunspots_df.to_csv(output_path, index=False)

def save_image(vertices, centroids, fits_image, output_path):
    # Convert image to 0-255
    filepath = f"{output_path.rsplit('.', 1)[0]}.png"
    plt.imsave(filepath, fits_image, cmap='gray', vmin=np.nanmin(fits_image), vmax=np.nanmax(fits_image))
    image = cv2.imread(filepath)

    # Draw vertices on cropped image as blue pixels
    for i in range(len(vertices)):
        image[vertices[i][1]][vertices[i][0]] = [255, 0, 0]
    
    # Draw centroid on cropped image as a red pixel
    for i in range(len(centroids)):
        image[centroids[i][1]][centroids[i][0]] = [0, 0, 255]

    # Save image
    print(f"Saving {filepath}")
    cv2.imwrite(filepath, image)

# ==========================================================================================
# Sunspot calculations
# ==========================================================================================
def get_sunspot_metrics(sunspot, image, output_path, threshold, min_conv_sum, max_conv_sum):
    offset_x = int(sunspot['x'])
    offset_y = int(sunspot['y'])
    width = int(sunspot["width"])
    height = int(sunspot["height"])
    
    # Crop image
    cropped_image = image[offset_y:offset_y + height, offset_x:offset_x + width]

    # Binarize image cutoff by threshold, normalize and invert
    sunspot_arr, min_value, max_value = binarize_image(cropped_image, threshold, width)

    # Find sunspot area
    area = np.count_nonzero(sunspot_arr)

    # Find average intensity within the sunspot
    average_intensity = find_average_intensity(sunspot_arr, cropped_image, width, height)

    # Find centroid
    centroid_x, centroid_y, vertices = find_centroid(sunspot_arr, min_conv_sum, max_conv_sum, width, height)

    # Find intensity of the centroid
    centroid_intensity = cropped_image[int(centroid_y), int(centroid_x)]

    # Print sunspot
    np.set_printoptions(precision=2, linewidth=200)
    print(f"Sunspot {offset_x}, {offset_y}")

    return area, average_intensity, offset_x + centroid_x, offset_y + centroid_y, min_value, max_value, centroid_intensity, vertices

def binarize_image(image, threshold, width):
    # List single values from image
    data = []
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            data.append(image[x, y])

    # Min-max normalize into percentages
    min_value = min(data)
    max_value = max(data)
    print(f"Min/Max values: {min_value}, {max_value}")

    for i in range(len(data)):
        data[i] = (data[i] - min_value) / (max_value - min_value)

    # Invert array
    for i in range(len(data)):
        data[i] = 1 - data[i]

    # Threshold cut-off
    for i in range(len(data)):
        if data[i] < threshold:
            data[i] = 0
        else:
            data[i] = 1

    # Return 2D array
    return np.reshape(data, (-1, width)), min_value, max_value

def find_centroid(sunspot_arr, min_conv_sum, max_conv_sum, w, h):
    vertices = []

    # Find vertices by summing adjacent elements using convolution
    h_hv_filter = np.array([ # weighted sum
        [0.25, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.50, 0.50, 0.50, 0.25],
        [0.25, 0.50, 1.00, 0.50, 0.25],
        [0.25, 0.50, 0.50, 0.50, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25]
    ])
    convolved_arr = scipy.signal.convolve2d(sunspot_arr, h_hv_filter, mode='same')

    # Print convolved array
    print(sunspot_arr)
    np.set_printoptions(formatter={'float': '{: 0.1f}'.format}, linewidth=100000)
    print(convolved_arr)

    # Find vertices in convolved array between 1 and min & max adjacent elements
    for y in range(h):
        for x in range(w):
            if convolved_arr[y, x] >= min_conv_sum and convolved_arr[y, x] <= max_conv_sum:
                vertices.append((x, y))

    # Find centroid position
    # Count vertices and denote number by n. Sum x & y values from vertices and divide by n
    n = len(vertices)
    if n > 0:
        sum_x, sum_y = [ sum(row[i] for row in vertices) for i in range(len(vertices[0])) ]
        centroid_x = sum_x / n
        centroid_y = sum_y / n
    else:
        centroid_x = 0
        centroid_y = 0

    return centroid_x, centroid_y, vertices

def find_average_intensity(sunspot_arr, image, w, h):
    vals = []
    for y in range(h):
        for x in range(w):
            if sunspot_arr[y, x] > 0:
                vals.append(image[y, x])
    
    average_intensity = sum(vals) / len(vals)

    return average_intensity

# ==========================================================================================
# Entry
# ==========================================================================================
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
