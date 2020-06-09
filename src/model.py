'''
Class to manipulate models used in gaze pointer controller.
'''

import os
import sys
import logging as log
import utils
from openvino.inference_engine import IENetwork, IECore

FACE_DETECTION_MODEL = 'intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml'
HEAD_POSE_MODEL = 'intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml'
FACE_LANDMARKS_MODEL = 'intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml'


class GenericModel:
    '''
    Class for controlling similar model characteristics.
    '''
    def __init__(self, device='CPU'):
        self.device = device
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        self.outputs = None


    def load_model(self, model, cpu_extension=None, plugin=None):
        # Obtain model files path:
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        if not plugin:
            log.info("Initializing plugin for {} device...".format(self.device))
            self.plugin = IECore()
        else:
            self.plugin = plugin

        if cpu_extension and 'CPU' in self.device:
            self.plugin.add_extension(cpu_extension, "CPU")

        log.info("Reading IR...")
        self.net = IENetwork(model=model_xml, weights=model_bin)

        log.info("Loading IR to the plugin...")

        # If applicable, add a CPU extension to self.plugin
        if "CPU" in self.device:
            supported_layers = self.plugin.query_network(self.net, "CPU")
            not_supported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]

            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(self.device,', '.join(not_supported_layers)))
                log.error("Please try to specify another cpu extension library path (via -l or --cpu_extension command line parameters)"
                          " that support required model layers or try, in last case, with other model")
                sys.exit(1)

        # Load the model to the network:
        self.net_plugin = self.plugin.load_network(network=self.net, device_name=self.device)

        # Obtain other relevant information:
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        return self.plugin

    def predict(self, image, request_id=0):
        '''
        Function to make an async inference request.
        '''
        preprocessed_image = self.preprocess_input(image)
        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, 
                                                                inputs={self.input_blob: preprocessed_image})

        return self.net_plugin

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Function to preprocess input image according to model requirement.
        '''

        input_shape = self.net.inputs[self.input_blob].shape
        preprocessed_image = utils.handle_image(image, input_shape[3], input_shape[2])

        return preprocessed_image

    def wait(self, request_id=0):
        status = self.net_plugin.requests[request_id].wait(-1)

        return status

    def get_output(self, request_id=0):

        self.outputs = self.net_plugin.requests[request_id].outputs[self.out_blob]

        return self.outputs


class FaceDetector(GenericModel):

    def __init__(self, model_path=FACE_DETECTION_MODEL):
        super().__init__()
        self.load_model(model_path)
    
    def preprocess_output(self):
        '''
        [1,1,N,7],  [image_id, label, conf, x_min, y_min, x_max, y_max]
        '''

        return self.outputs[0,0]

class HeadPose(GenericModel):

    def __init__(self, model_path=HEAD_POSE_MODEL):
        super().__init__()
        self.load_model(model_path)

    def get_output(self, request_id=0):
        '''
        Por ahora devolviendo un diccionario, depende de como se vaya a manejar la lógica después se modifica preprocess_output.
        '''
        self.outputs = self.net_plugin.requests[request_id].outputs

        return self.outputs

    
    def preprocess_output(self):
        '''
        Output layer names in Inference Engine format:

    name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
    name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
    name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).

        '''
        return self.outputs['angle_p_fc'][0,0], self.outputs['angle_r_fc'][0,0], self.outputs['angle_y_fc'][0,0]
        #return None

class FaceLandmarks(GenericModel):
    def __init__(self, model_path=FACE_LANDMARKS_MODEL):
        super().__init__()
        self.load_model(model_path)
    
    def preprocess_output(self):
        '''
        The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 
        floating point values for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5). 
        All the coordinates are normalized to be in range [0,1].
        '''
        # out = self.outputs[0,:,0,0]

        # length = len(out)
        # output = []
        # for i in range(length-1):
        #     x = out[i]
        #     y = out[i+1]


        return self.outputs[0,:,0,0]


######
# import cv2
# #img = cv2.imread('imagen.jpg')
# cap = cv2.VideoCapture(0)

# while(cap.isOpened()):
#     ret, img = cap.read()
#     # img2 = img.copy()
#     facedetector = FaceDetector()
#     facedetector.predict(img)
#     facedetector.wait()
#     facedetector.get_output()
#     out = facedetector.preprocess_output()
#     detection = []
#     for o in out:
#         if o[2] > 0.6:
#             xmin = o[3]
#             ymin = o[4]
#             xmax = o[5]
#             ymax = o[6]
#             detection.append([xmin, ymin, xmax, ymax])

#     w = img.shape[1]
#     h = img.shape[0]
#     detection = detection[0]
#     detection = [detection[0]*w, detection[1]*h, detection[2]*w, detection[3]*h]

#     img2 = img[int(detection[1]):int(detection[3]), int(detection[0]):int(detection[2])]
#     facelm = FaceLandmarks()
#     pred = facelm.predict(img2)
#     facelm.wait()
#     facelm.get_output()
#     out = facelm.preprocess_output()
#     img = utils.draw_points(img, out[:8], thickness=8)

#     # headpose = HeadPose()
#     # headpose.predict(img)
#     # headpose.wait()
#     # headpose.get_output()
#     # print('headpose',headpose.preprocess_output())

    # cv2.imshow('frame',img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
# # cv2.imwrite('resultado.png', img)
