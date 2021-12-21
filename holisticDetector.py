# https://www.youtube.com/watch?v=EgjwKM3KzGU

import mediapipe as mp
import cv2

from google.protobuf.json_format import MessageToDict

# Initializations: static code
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


class HolisticDetector:
    def __init__(self, min_detection_confidence=0.9, min_tracking_confidence=0.7):
        # when the mediapipe is first started, it detects the hands. After that it tries to track the hands
        # as detecting is more time consuming than tracking. If the tracking confidence goes down than the
        # specified value then again it switches back to detection
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def findLandMarks(self, image):

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.holistic.process(image)

        return results

    def drawLeftLandMarks(self, image, results):
        # Draw landmark annotation on the image.
        image.flags.writeable = True

        # mp_drawing.draw_landmarks(
        #     image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        return image
