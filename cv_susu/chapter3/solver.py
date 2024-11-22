import cv2
import numpy as np

class Solver(object):
    def __init__(self, calibration_image_rgb: np.ndarray):
        self.calibration_image_rgb = calibration_image_rgb
        

        calibration_image_ycrcb = cv2.cvtColor(self.calibration_image_rgb, cv2.COLOR_RGB2YCrCb)
        
        self.lower_skin = np.array([13, 120, 75], dtype=np.uint8)
        self.upper_skin = np.array([255, 195, 122], dtype=np.uint8)

    def solve(self, image_rgb: np.ndarray):
        image_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        
        skin_mask = cv2.inRange(image_ycrcb, self.lower_skin, self.upper_skin)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        return skin_mask