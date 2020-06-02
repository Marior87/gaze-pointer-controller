import numpy as np
import cv2


class Person():
    """
    Class to manage detections instances.
    """

    def __init__(self, id, frame_init):
        self.id = id
        self.frame_init = frame_init


def handle_image(input_image, width=60, height=60):
    """
    Function to preprocess input image and return it in a shape accepted by the model.
    Default arguments are set for facial landmark model requirements.
    """
    preprocessed_image = cv2.resize(input_image, (width, height))
    preprocessed_image = preprocessed_image.transpose((2,0,1))
    preprocessed_image = preprocessed_image.reshape(1, 3, height, width)

    return preprocessed_image


def calculate_centroid(bounding_box):
    """
    Function to calculate centroid relative position (x,y) given a bounding box.
    """

    xmin = bounding_box[0]
    ymin = bounding_box[1]
    xmax = bounding_box[2]
    ymax = bounding_box[3]

    return ((xmin+xmax)/2, (ymin+ymax)/2)


def draw_bounding_box(frame, detection, color =(0,255,0)):
    """
    Function to draw a bounding box over a frame on a detection.

    Args:
        frame: A frame (or image) where to draw the bounding boxes.
        detection: A list of a bounding box coordinates.
        color: Bounding box color (in BGR).

    Returns:
        img: Image with all the bounding boxes drawed.
    """

    # Obtain current frame's dimentions:
    height = frame.shape[0]
    width = frame.shape[1]

    # Loop over every detection:
    xmin = int(detection[0]*width)  # Obtain Top Left X coordinate.
    ymin = int(detection[1]*height) # Obtain Top Left Y coordinate.
    xmax = int(detection[2]*width)  # Obtain Bottom Right X coordinate.
    ymax = int(detection[3]*height) # Obtain Bottom Right Y coordinate.

    # Draw the bounding box:
    img = cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color=color,thickness=5)

    return img


def draw_text(frame, 
                text, 
                coordinates=(0.1,0.1), 
                font = cv2.FONT_HERSHEY_SIMPLEX, 
                font_size = 0.7, 
                font_color = (0,0,0), 
                font_thickness = 2):
    """
    Function to draw text over a bounding box.

    Args:
        frame: Frame (or image) where to draw text.
        texts: A list of text, should be the same lenght as the quantity of bounding boxes.
        results_bb: A list of bounding box coordinates, which each one is in the form (xmin, ymin, xmax, ymax).
        font: OpenCV font style
        font_size: OpenCV font size.
        font_color: BGR font color tuple.
        font_thickness: Font line thickness.
        offset_x: Offset in x position of text regarding the upper left corner
        offset_y: Offset in y position of text regarding the upper left corner

    Returns:
        img: Image with required text drawn.
    """

    #assert len(results_bb) == len(texts), "Number of bounding boxes ({}) not the same as number of texts ({})".format([len(results_bb), len(texts)])
    
    h = frame.shape[0]
    w = frame.shape[1]

    img = frame.copy()

    x = int(coordinates[0]*w)
    y = int(coordinates[1]*h)

    img = cv2.putText(img, text,(x,y), font, font_size, font_color, font_thickness)

    return img
