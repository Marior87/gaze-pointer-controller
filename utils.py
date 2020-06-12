import numpy as np
import cv2


def handle_image(input_image, width=60, height=60):
    """
    Function to preprocess input image and return it in a shape accepted by the model.
    Default arguments are set for facial landmark model requirements.
    """
    preprocessed_image = cv2.resize(input_image, (width, height))
    preprocessed_image = preprocessed_image.transpose((2,0,1))
    preprocessed_image = preprocessed_image.reshape(1, 3, height, width)

    return preprocessed_image


def get_eyes_crops(face_crop, right_eye, left_eye, relative_eye_size=0.20):

    crop_w = face_crop.shape[1]
    crop_h = face_crop.shape[0]

    x_right_eye = right_eye[0]*crop_w
    y_right_eye = right_eye[1]*crop_h
    x_left_eye = left_eye[0]*crop_w
    y_left_eye = left_eye[1]*crop_h

    relative_eye_size_x = crop_w*relative_eye_size
    relative_eye_size_y = crop_h*relative_eye_size

    right_eye_dimensions = [int(y_right_eye-relative_eye_size_y/2), int(y_right_eye+relative_eye_size_y/2),
    int(x_right_eye-relative_eye_size_x/2), int(x_right_eye+relative_eye_size_x/2)]

    left_eye_dimensions = [int(y_left_eye-relative_eye_size_y/2), int(y_left_eye+relative_eye_size_y/2),
    int(x_left_eye-relative_eye_size_x/2), int(x_left_eye+relative_eye_size_x/2)]

    right_eye_crop = face_crop[right_eye_dimensions[0]:right_eye_dimensions[1], 
                                right_eye_dimensions[2]:right_eye_dimensions[3]]

    left_eye_crop = face_crop[left_eye_dimensions[0]:left_eye_dimensions[1],
                                left_eye_dimensions[2]:left_eye_dimensions[3]]

    return right_eye_crop, left_eye_crop, right_eye_dimensions, left_eye_dimensions