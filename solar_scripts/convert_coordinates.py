# ==========================================================================================
# Convert relative coordinates from Yolov5 to pixels
# Maintainer: Christopher Toumanian
# Email: cct_580@usc.edu
# ==========================================================================================
import csv
from argparse import ArgumentParser

# ==========================================================================================
# Arguments
# Example: python convert_coordinates.py -f runs/detect/exp/labels/bus.txt -o runs/detect/exp/labels/bus.csv -w 810 -h 1080 -v -H -d ","
# ==========================================================================================
def parse_arguments():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("-f", "--file", dest="input_filepath", help="File to read", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output_filepath", help="File to write", metavar="FILE")
    parser.add_argument("-w", "--width", dest="image_width", help="Width of image", metavar="N")
    parser.add_argument("-h", "--height", dest="image_height", help="Height of image", metavar="N")
    parser.add_argument("-d", "--delimiter", dest="delimiter", help="Delimiter of CSV output file", metavar="S")
    parser.add_argument("-H", "--header", dest="header", help="Include header in CSV output file", action="store_true")
    parser.add_argument("-v", "--verbose", dest="verbose", help="Verbose output", action="store_true")
    args = parser.parse_args()
    return args

# ==========================================================================================
# Classes
# ==========================================================================================
class Feature:
    def __init__(self, **args):
        self.label = args.get('label')
        self.confidence = args.get('confidence')
        self.x_center = 0
        self.y_center = 0
        self.x_center_relative = args.get('x_center_relative')
        self.y_center_relative = args.get('y_center_relative')
        self.width = 0
        self.height = 0
        self.width_relative = args.get('width_relative')
        self.height_relative = args.get('height_relative')
        self.x_1 = 0
        self.y_1 = 0
        self.x_2 = 0
        self.y_2 = 0

    def __iter__(self):
        return iter([
            self.label,
            self.confidence,
            self.x_center_relative,
            self.y_center_relative,
            self.width_relative,
            self.height_relative,
            self.x_center,
            self.y_center,
            self.width,
            self.height,
            self.x_1,
            self.y_1,
            self.x_2,
            self.y_2
        ])

    def convert_relative_coordinates(self, image_dimensions):
        relative_center_coordinate = (self.x_center_relative, self.y_center_relative)
        relative_bbox_dimensions = (self.width_relative, self.height_relative)

        self.x_center, self.y_center = self.to_pixel_coords(relative_center_coordinate, image_dimensions)
        self.width, self.height = self.to_pixel_coords(relative_bbox_dimensions, image_dimensions)
        self.x_1 = self.x_center - int(self.width/2)
        self.y_1 = self.y_center - int(self.height/2)
        self.x_2 = self.x_center + int(self.width/2)
        self.y_2 = self.y_center + int(self.height/2)

    @staticmethod
    def to_pixel_coords(relative_coords, image_dimensions):
        return tuple(round(coord * dimension) for coord, dimension in zip(relative_coords, image_dimensions))

# ==========================================================================================
# Reader
# ==========================================================================================
def get_features_from_file(filepath):
    features = []

    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            attributes = line.split()
            feature = Feature(
                label=int(attributes[0]),
                x_center_relative=float(attributes[1]),
                y_center_relative=float(attributes[2]),
                width_relative=float(attributes[3]),
                height_relative=float(attributes[4]),
                confidence=float(attributes[5])
            )
            features.append(feature)
            
    return features

# ==========================================================================================
# Writer
# ==========================================================================================
def write_features_to_file(features, filepath, delimiter, header):
    with open(filepath, "w") as stream:
        writer = csv.writer(stream, delimiter=delimiter)

        if header:
            attribute_names = [
                "label",
                "confidence",
                "x_center_relative",
                "y_center_relative",
                "width_relative",
                "height_relative",
                "x_center",
                "y_center",
                "width",
                "height",
                "x_1",
                "y_1",
                "x_2",
                "y_2"
            ]
            writer.writerow(attribute_names)

        writer.writerows(features)

# ==========================================================================================
# Main
# ==========================================================================================
def main():
    args = parse_arguments()

    # Step 1: Read Yolov5 label file with label, relative coordinates, and confidences
    # Ex: label, x_center, y_center, width, height, confidence
    #     1 0.0419753 0.660648 0.0839506 0.299074 0.535157
    if args.verbose:
        print(f"Reading file: {args.input_filepath}")

    features = get_features_from_file(args.input_filepath)

    # Step 2: Convert relative coordinates to pixel coordinates
    if args.verbose:
        print(f"Converting to bounding boxes with pixel coordinates...")
        print("label confidence x_center y_center width height x_1 y_1 x_2 y_2")

    for feature in features:
        feature.convert_relative_coordinates((int(args.image_width), int(args.image_height)))

        if args.verbose:
            print(
                feature.label,
                feature.confidence,
                feature.x_center,
                feature.y_center,
                feature.width,
                feature.height,
                feature.x_1,
                feature.y_1,
                feature.x_2,
                feature.y_2
            )

    # Step 3: Write to file
    if args.verbose:
        print(f"Writing to file: {args.output_filepath} with '{args.delimiter}' as delimiter.")

    write_features_to_file(features, args.output_filepath, args.delimiter, args.header)

    if args.verbose:
        print("Done.")

if __name__ == "__main__":
    main()
