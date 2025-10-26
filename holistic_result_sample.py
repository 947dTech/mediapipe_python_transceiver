#!/usr/bin/env python3

# holistic_trackingのresultsがどんなデータを持っているか検証
# https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md

import cv2
import mediapipe as mp
import numpy as np

from utils import holistic_results_to_dict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


# For static images:
def holistic(file):
    with mp_holistic.Holistic(static_image_mode=True,
                              model_complexity=2,
                              enable_segmentation=True,
                              refine_face_landmarks=True) as holistic:
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
            )

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (192, 192, 192)
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose, left and right hands, and face landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_pose_landmarks_style())
        # cv2.imwrite('/tmp/annotated_image.png', annotated_image)
        cv2.imshow('annotated_image', annotated_image)
        # Plot pose world landmarks.
        # mp_drawing.plot_landmarks(
        #     results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

        # resultsの中身をjson化する
        json_dict = {}
        json_dict = holistic_results_to_dict(results, json_dict)

        return results, json_dict


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    results, json_dict = holistic(args.input)

    print(json_dict)
    cv2.waitKey()
