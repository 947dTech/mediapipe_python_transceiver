#!/usr/bin/env python3

# holistic_trackingのandroid改造版と同じデータを出力する
# https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md
# holisticのサンプルコードが現状ないので各landmarkerを参考に作成
# https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
# https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
# https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

from argparse import ArgumentParser
import json
import math
import os
import subprocess
import socket
import time

import cv2
import mediapipe as mp

from utils import holistic_results_to_dict

from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    RunningMode,
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
    running_mode=RunningMode.VIDEO,
    output_face_blendshapes=True,
    output_segmentation_mask=True,
    )

parser = ArgumentParser()
parser.add_argument("-i", "--input", default=0)
parser.add_argument("-d", "--device", default="")
parser.add_argument("-r", "--rate", default=30)
parser.add_argument("--width", default=1920)
parser.add_argument("--height", default=1080)
parser.add_argument("--codec", default="MJPG")
parser.add_argument("-c", "--calib_file", default="cameraParameters.xml")
parser.add_argument("--host", default="192.168.11.19")
parser.add_argument("--port", default=0x947d)

args = parser.parse_args()

# カメラデバイスはdeviceが指定された場合そちらを優先、開けなければその時点で中断
if args.device == "":
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] cannot open id {args.input}")
        exit(1)
else:
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"[ERROR] cannot open device {args.device}")
        exit(1)

# webカメラの設定は個体ごとに異なるため要確認
ret = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.codec))
if not ret:
    print(f"[WARN] cannot set prop FOURCC = {args.codec}")
ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
if not ret:
    print(f"[WARN] cannot set prop WIDTH = {args.width}")
ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
if not ret:
    print(f"[WARN] cannot set prop HEIGHT = {args.height}")
ret = cap.set(cv2.CAP_PROP_FPS, float(args.rate))
if not ret:
    print(f"[WARN] cannot set prop RATE = {args.rate}")

# キャリブレーションの適用
# TODO: キャリブレーションされていなくても先に進めるようにしたほうがいい
calib_file = args.calib_file
fs = cv2.FileStorage(calib_file, cv2.FileStorage_READ)
calib_resolution = fs.getNode("cameraResolution")
calib_width = calib_resolution.at(0).real()
calib_height = calib_resolution.at(1).real()
camera_matrix = fs.getNode("cameraMatrix").mat()
dist_coeffs = fs.getNode("dist_coeffs").mat()

aspect_ratio = float(args.width) / float(args.height)
calib_aspect_ratio = calib_width / calib_height
aspect_eq = math.fabs(calib_aspect_ratio - aspect_ratio) < 1e-3
if not aspect_eq:
    print("mismatch aspect ratio:")
    print(f"For camera device: {args.width} x {args.height}")
    print(f"In {calib_file}: {calib_width} x {calib_height}")
    exit(1)

if args.width != int(calib_width):
    print("scale camera matrix")
    print(f"For camera device: {args.width} x {args.height}")
    print(f"In {calib_file}: {calib_width} x {calib_height}")
    image_scale = float(args.width) / calib_width
    camera_matrix *= image_scale
    camera_matrix[2, 2] = 1.0

calibrated = True

# UDPソケットの準備
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# TODO: udp自体の仕様上、65535を超えることはできないようだが、かなり大きな値を入れないと変更できていない
# print("buffersize: ", sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)
max_buf_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
# print("buffersize: ", sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))

prev_timestamp = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

with HolisticLandmarker.create_from_options(options) as holistic:
    while cap.isOpened():
        success, rawimage = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # 画像を取得した時点のタイムスタンプを保持
        timestamp = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        timestamp_ms = int(timestamp / 1e6)

        # キャリブレーションの適用
        if calibrated:
            image = cv2.undistort(rawimage, camera_matrix, dist_coeffs)
        else:
            image = rawimage

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=image),
            timestamp_ms)

        # udp送信するデータ
        json_dict = {}

        # 解像度と焦点距離、キャリブレーション必須
        json_dict["camera_params"] = {
            "focal_length": camera_matrix[0, 0],
            "cx": camera_matrix[0, 2],
            "cy": camera_matrix[1, 2],
            "frame_width": image.shape[1],
            "frame_height": image.shape[0]
        }

        # 重力方向、android端末に準拠、Y up X right Z front
        json_dict["gravity"] = [0.0, 9.80665, 0.0]
        json_dict["gravity_stamp"] = timestamp

        # resultsの中身を追加
        json_dict = holistic_results_to_dict(results, json_dict, timestamp)

        # UDP送信
        # json_msg = json.dumps(json_dict, ensure_ascii=False).encode("UTF-8")
        # print("message size: ", len(json_msg))
        # 有効数字の桁数を変える方法、Androidで送信サイズを削るために同じことをやっている
        json_msg = json.dumps(json.loads(json.dumps(json_dict), parse_float=lambda x: round(float(x), 3))).encode("UTF-8")
        print("message size: ", len(json_msg))

        # バッファサイズに収まらない場合送信しない、現状確認できていない
        if (len(json_msg) > max_buf_size):
            print("message size exceeded from max buffer size, skip.")
        else:
            sock.sendto(json_msg, (args.host, args.port))

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(1) & 0xFF == 27:
            # 27: escape
            break

        print("fps: ", 1e9 / (timestamp - prev_timestamp))
        prev_timestamp = timestamp
cap.release()
