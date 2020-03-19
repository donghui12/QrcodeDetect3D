import copy

import cv2
import numpy as np
import time
from QrcodeDetect3D import QRCode

# # Length of pattern in realily
QRCodeSide = 8.8
# Pixel length of pattern in image
PatternSide = 250


class QRCodeDetector(object):
    def __init__(self):
        """
        init camera
        """
        self.cam_no = 1
        width = 1920
        height = 1080
        self.cam = cv2.VideoCapture(self.cam_no)
        self.cam.set(3, width)  # 设置帧宽
        self.cam.set(4, height)  # 设置帧高

        # Length of pattern in realily
        QRCodeSide = 8.8
        # Pixel length of pattern in image
        PatternSide = 250

    def get_frame(self):
        img = cv2.flip(self.cam.read()[1], 1)
        return img

    @staticmethod
    def camPoseEstimate(frame):
        # OpenCV on Mac OSX has some issue on image size swapping while reading
        # Mac user might need this line to rotate the image by 90 degree clockwise
        # image = cv2.flip(cv2.transpose(frame), 1)
        image = copy.deepcopy(frame)

        size = image.shape

        # Pattern points in 2D image coordinates
        image_array = QRCode.detectQRCode(image)
        if image_array is None:
            return None, 0
        pattern_points = np.array(image_array, dtype='double')

        # Pattern points in 3D world coordinates.
        model_points = np.array([(-QRCodeSide / 2, QRCodeSide / 2, 0.0),
                                 (QRCodeSide / 2, QRCodeSide / 2, 0.0),
                                 (QRCodeSide / 2, -QRCodeSide / 2, 0.0),
                                 (-QRCodeSide / 2, -QRCodeSide / 2, 0.0),
                                 ])

        focal_length = size[1]
        camera_center = (size[1] / 2, size[0] / 2)

        # Initialize approximate camera intrinsic matrix
        camera_intrinsic_matrix = np.array([[focal_length, 0, camera_center[0]],
                                            [0, focal_length, camera_center[1]],
                                            [0, 0, 1]
                                            ], dtype="double")

        # Assume there is no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Get camera extrinsic matrix - R and T
        flag, rotation_vector, translation_vector = cv2.solvePnP(model_points,
                                                                 pattern_points,
                                                                 camera_intrinsic_matrix,
                                                                 dist_coeffs,
                                                                 flags=cv2.SOLVEPNP_ITERATIVE)

        # Convert 3x1 rotation vector to rotation matrix for further computation
        rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

        # C = -R.transpose() * T
        C = np.matmul(-rotation_matrix.transpose(), translation_vector)

        # Orientation vector
        O = np.matmul(rotation_matrix.T, np.array([0, 0, 1]).T)

        return C.squeeze(), O

    def get_QRCode(self):
        """
        	Visualize 3D model with Matplotlib 3D.
        	Input:
        		image_path: Input image path - string
        	Output:
        		None -
        """
        location_point = [0, 0, 0]
        frame = self.get_frame()
        camera_pose, camera_orientation = self.camPoseEstimate(frame)
        if camera_pose is not None:
            location_point = [int(i) for i in camera_pose]
        return location_point


def test_QRCodeDetector():
    a = QRCodeDetector()
    while True:
        frame_init = a.get_frame()
        cv2.imshow("img", frame_init)
        result = None

        result = a.get_QRCode()
        if any(result):
            print(result)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


test_QRCodeDetector()
