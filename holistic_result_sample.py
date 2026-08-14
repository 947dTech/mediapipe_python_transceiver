#!/usr/bin/env python3

# holistic_trackingのresultsがどんなデータを持っているか検証
# https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md
# holisticのサンプルコードが現状ないので各landmarkerを参考に作成
# https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
# https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
# https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess

from utils import holistic_results_to_dict

from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarksConnections,
    PoseLandmark,
    PoseLandmarksConnections,
    HandLandmarksConnections,
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    HolisticLandmarkerResult
)

# taskファイルのURLは以下から
# https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/holistic-landmarker.ts#L113

task_file = 'models/holistic_landmarker.task'
task_file_url = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task'
if not os.path.exists(task_file):
    subprocess.run(['wget', '-O', task_file, task_file_url])

base_options = BaseOptions(
    model_asset_path=task_file,
    # delegate=BaseOptions.Delegate.GPU
    )
options = HolisticLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_segmentation_mask=True,
    )


# For static images:
def holistic(file, msg_mode="legacy"):
    with HolisticLandmarker.create_from_options(options) as holistic:
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = holistic.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

        if results.pose_landmarks:
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks[PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks[PoseLandmark.NOSE].y * image_height})'
            )

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.squeeze(
            np.stack(
                (results.segmentation_mask.numpy_view(),) * 3, axis=-1) > 0.1)
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (192, 192, 192)
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose, left and right hands, and face landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
        # cv2.imwrite('/tmp/annotated_image.png', annotated_image)
        cv2.imshow('annotated_image', annotated_image)
        # Plot pose world landmarks.
        # mp_drawing.plot_landmarks(
        #     results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

        # resultsの中身を確認
        print('results:')
        for attr in vars(results):
            val = getattr(results, attr)
            if isinstance(val, list):
                print(f'  {attr}: {len(val)}')
            else:
                print(f'  {attr}')

        # resultsの中身をjson化する
        hand2d = True
        hand3d = True
        legacy_face_mode = False
        face_blendshapes = True
        if msg_mode == "legacy":
            hand3d = False
            legacy_face_mode = True
            face_blendshapes = False
        elif msg_mode == "latest":
            hand2d = False

        json_dict = {}
        json_dict = holistic_results_to_dict(results, json_dict,
                                             hand2d=hand2d, hand3d=hand3d,
                                             legacy_face_mode=legacy_face_mode,
                                             face_blendshapes=face_blendshapes)

        return results, json_dict


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--msg-mode", default="legacy",
                        choices=["legacy", "latest", "full"])
    args = parser.parse_args()

    results, json_dict = holistic(args.input, args.msg_mode)

    print(json_dict)
    print('json:')
    landmark_names = [
        'pose_landmarks',
        'pose_world_landmarks',
        'face_landmarks',
        'face_blendshapes',
        'right_hand_landmarks',
        'right_hand_world_landmarks',
        'left_hand_landmarks',
        'left_hand_world_landmarks',
    ]
    for landmark_name in landmark_names:
        if landmark_name in json_dict:
            print(f'  {landmark_name}: {len(json_dict[landmark_name])}')
        else:
            print(f'  {landmark_name}: not included')

    cv2.waitKey()
