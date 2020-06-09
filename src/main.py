'''Script to run the application'''

from model import FaceDetector, FaceLandmarks, HeadPose
from input_feeder import InputFeeder
from mouse_controller import MouseController
from argparse import ArgumentParser

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file. Use CAM to use webcam stream")
    
    # Optional arguments:
    parser.add_argument("-mf", "--model_facedetector", required=False, type=str, default=None,
                        help="Path to an xml file with a trained face detector model.")
    parser.add_argument("-ml", "--model_facelm", required=False, type=str, default=None,
                        help="Path to an xml file with a trained face landmarks detector model.")
    parser.add_argument("-mh", "--model_headpose", required=False, type=str, default=None,
                        help="Path to an xml file with a trained head pose detector model.")
                                                                
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
    return parser

# Function to instantiate and return models:
def get_models(args):

    model_facedetector = args.model_facedetector
    model_facelm = args.model_facelm
    model_headpose = args.model_headpose

    # Get face detector model:
    if model_facedetector:
        facedetector = FaceDetector(model_path=model_facedetector)
    else:
        facedetector = FaceDetector()
    
    # Get face landmarks detector model:
    if model_facelm:
        facelm = FaceDetector(model_path=model_facelm)
    else:
        facelm = FaceLandmarks()
    
    # Get headpose detector model:
    if model_headpose:
        headpose = FaceDetector(model_path=model_headpose)
    else:
        headpose = HeadPose()
    
    return facedetector, facelm, headpose

# Function to return a crop of a detected face:
def get_face_crop(frame, facedetector, args):

    threshold = args.prob_threshold
    output = None

    # Obtain face coordinates:
    try:
        facedetector.predict()
        facedetector.wait()
        facedetector.get_output()
        output = facedetector.preprocess_output()
    except:
        return None

    if output:
        detection = []
        for o in output:
            if o[2] > threshold:
                xmin = o[3]
                ymin = o[4]
                xmax = o[5]
                ymax = o[6]
                detection.append([xmin, ymin, xmax, ymax])

    # I will only use one detection, as only one face can control the pointer
    detection = detection[0]

    # Converting relative coordinates to absolute coordinates:
    w = frame.shape[1]
    h = frame.shape[0]
    detection = [detection[0]*w, detection[1]*h, detection[2]*w, detection[3]*h]

    return frame[int(detection[1]):int(detection[3]), int(detection[0]):int(detection[2])]

# Function to get eyes coordinates:
def get_eyes_coordinates(face_crop, facelm):

    output = None

    try:
        facelm.predict(face_crop)
        facelm.wait()
        facelm.get_output()
        output = facelm.preprocess_output()
    except:
        return None

    if output:
        

    return None

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()


if __name__ == '__main__':
    main()