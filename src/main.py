'''Script to run the application'''

import cv2
import numpy as np
import utils
import logging as log
import time
import csv

from model import FaceDetector, FaceLandmarks, HeadPose, Gaze
from mouse_controller import MouseController
from argparse import ArgumentParser

log.basicConfig(level=log.DEBUG)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file. Use CAM to use webcam stream")
    
    # Optional arguments:

    # Models:
    parser.add_argument("-mf", "--model_facedetector", required=False, type=str, default=None,
                        help="Path to an xml file with a trained face detector model.")
    parser.add_argument("-ml", "--model_facelm", required=False, type=str, default=None,
                        help="Path to an xml file with a trained face landmarks detector model.")
    parser.add_argument("-mh", "--model_headpose", required=False, type=str, default=None,
                        help="Path to an xml file with a trained head pose detector model.")
    parser.add_argument("-mg", "--model_gaze", required=False, type=str, default=None,
                        help="Path to an xml file with a trained gaze detector model.")

    # Models handlers:                                                            
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-fc", "--frame_count", type=float, default=20,
                        help="How many frames count to actually run the models."
                        "(20 by default)")

    # Visualization:
    parser.add_argument("-vf", "--view_face", type=str, default=False,
                    help="Flag to indicate that user wants to see face bounding box")

    parser.add_argument("-ve", "--view_eyes", type=str, default=False,
                    help="Flag to indicate that user wants to see eyes bounding boxes")

    parser.add_argument("-vh", "--view_headpose", type=str, default=False,
                    help="Flag to indicate that user wants to see headpose angles")

    parser.add_argument("-vg", "--view_gaze", type=str, default=False,
                    help="Flag to indicate that user wants to see gaze direction")

    return parser

# Function to instantiate and return models:
def get_models(args):
    model_facedetector = args.model_facedetector
    model_facelm = args.model_facelm
    model_headpose = args.model_headpose
    model_gaze = args.model_gaze

    # Get face detector model:
    if model_facedetector:
        facedetector = FaceDetector(model_path=model_facedetector, device=args.device)
    else:
        facedetector = FaceDetector(device=args.device)
    
    # Get face landmarks detector model:
    if model_facelm:
        facelm = FaceLandmarks(model_path=model_facelm, device=args.device)
    else:
        facelm = FaceLandmarks(device=args.device)
    
    # Get headpose detector model:
    if model_headpose:
        headpose = HeadPose(model_path=model_headpose, device=args.device)
    else:
        headpose = HeadPose(device=args.device)

    if model_gaze:
        gaze = Gaze(model_path=model_gaze, device=args.device)
    else:
        gaze = Gaze()

    return facedetector, facelm, headpose, gaze


def gaze_pointer_controller(args, facedetector, facelm, headpose, gaze):

    mouse_controller = MouseController(precision='high', speed='fast')

    # Handle input type:
    inference_time_face = []
    inference_time_landmarks = []
    inference_time_headpose = []
    inference_time_gaze = []
    if args.input != 'CAM':
        try:
            # It seems that OpenCV can use VideoCapture to treat videos and images:
            input_stream = cv2.VideoCapture(args.input)
            length = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            webcamera = False

            # Check if input is an image or video file:
            if length > 1:
                single_image_mode = False
            else:
                single_image_mode = True

        except:
            print('Not supported image or video file format. Please pass a supported one.')
            exit()

    else:
        input_stream = cv2.VideoCapture(0)
        single_image_mode = False
        webcamera = True

    if not single_image_mode:
        count = 0
        while(input_stream.isOpened()):
        
            # Read the next frame:
            flag, frame = input_stream.read()

            if not flag:
                break

            if count % args.frame_count == 0:
                start = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # We get a detected face crop and its coordinates:
                face_crop, detection = facedetector.get_face_crop(frame, args)
                finish_face_detector_time = time.time()
                face_detector_time = round(finish_face_detector_time-start,5)
                log.info("Face detection took {} seconds.".format(face_detector_time))
                inference_time_face.append(face_detector_time)

                # Obtain eyes coordinates:
                right_eye, left_eye = facelm.get_eyes_coordinates(face_crop)

                # Obtain eyes crops:
                right_eye_crop, left_eye_crop, right_eye_coords, left_eye_coords = utils.get_eyes_crops(face_crop, right_eye,left_eye)
                finish_eyes_coordinates = time.time()
                eyes_detector_time = round(finish_eyes_coordinates-finish_face_detector_time,5)
                log.info("Eyes detection took {} seconds.".format(eyes_detector_time))
                inference_time_landmarks.append(eyes_detector_time)

                # Obtain headpose angles:
                headpose_angles = headpose.get_headpose_angles(face_crop)
                finish_headpose_angles = time.time()
                headpose_detector_time = round(finish_headpose_angles-finish_eyes_coordinates,5)
                log.info("Headpose angles detection took {} seconds.".format(headpose_detector_time))
                inference_time_headpose.append(headpose_detector_time)

                # Obtain gaze vector and mouse movement values:
                (x_movement, y_movement), gaze_vector = gaze.get_gaze(right_eye_crop, left_eye_crop, headpose_angles)
                finish_gaze_detection_time = time.time()
                gaze_detector_time = round(finish_gaze_detection_time-finish_headpose_angles,5)
                log.info("Gaze detection took {} seconds.".format(gaze_detector_time))
                inference_time_gaze.append(gaze_detector_time)


                # Optional visualization configuration:
                if args.view_face:
                    frame = cv2.rectangle(frame,
                                          (detection[0],detection[1]),
                                          (detection[2],detection[3]), 
                                          color=(0,255,0), 
                                          thickness=5)

                if args.view_eyes:

                    right_eye_coords = [right_eye_coords[0]+detection[1], right_eye_coords[1]+detection[1], 
                                right_eye_coords[2]+detection[0], right_eye_coords[3]+detection[0]]

                    left_eye_coords = [left_eye_coords[0]+detection[1], left_eye_coords[1]+detection[1], 
                                left_eye_coords[2]+detection[0], left_eye_coords[3]+detection[0]]
                    frame = cv2.rectangle(frame,
                                            (right_eye_coords[2],right_eye_coords[1]),
                                            (right_eye_coords[3],right_eye_coords[0]), 
                                            color=(255,0,0), 
                                            thickness=5)

                    frame = cv2.rectangle(frame,
                                            (left_eye_coords[2],left_eye_coords[1]),
                                            (left_eye_coords[3],left_eye_coords[0]), 
                                            color=(255,0,0), 
                                            thickness=5)

                if args.view_gaze:
                
                    # Right eye:
                    x_r_eye = int(right_eye[0]*face_crop.shape[1]+detection[0])
                    y_r_eye = int(right_eye[1]*face_crop.shape[0]+detection[1])
                    x_r_shift, y_r_shift = int(x_r_eye+gaze_vector[0]*100), int(y_r_eye-gaze_vector[1]*100)

                    # Left eye:
                    x_l_eye = int(left_eye[0]*face_crop.shape[1]+detection[0])
                    y_l_eye = int(left_eye[1]*face_crop.shape[0]+detection[1])
                    x_l_shift, y_l_shift = int(x_l_eye+gaze_vector[0]*100), int(y_l_eye-gaze_vector[1]*100)

                    frame = cv2.arrowedLine(frame, (x_r_eye, y_r_eye), (x_r_shift, y_r_shift), (0, 0, 255), 2)
                    frame = cv2.arrowedLine(frame, (x_l_eye, y_l_eye), (x_l_shift, y_l_shift), (0, 0, 255), 2)

                if args.view_headpose:

                    frame = cv2.putText(frame, 'Yaw: '+
                                    str(headpose_angles[0])+' '+
                                    'Pitch: '+str(headpose_angles[1])+' '+
                                    'Roll: '+str(headpose_angles[2]),(15,20),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2)

                # Resizing window for visualization convenience:
                cv2.namedWindow('Prueba',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Prueba', 600,400)
                cv2.imshow('Prueba', frame)

                mouse_controller.move(x_movement,y_movement)
            count = count + 1

        input_stream.release()
        with open('times.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Face Detector','Eyes Detector', 'Headpose Detector', 'Gaze Detector'])
            for i in range(len(inference_time_face)):
                writer.writerow([inference_time_face[i], inference_time_landmarks[i], inference_time_headpose[i], inference_time_gaze[i]])

    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # Get the models:
    facedetector, facelm, headpose, gaze = get_models(args)

    # Initiate:
    gaze_pointer_controller(args, facedetector, facelm, headpose, gaze)


if __name__ == '__main__':
    main()
