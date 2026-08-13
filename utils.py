# holisticの結果をjsonに変更する

import time


def landmark_to_dict(landmark):
    '''landmarkをdictに変換する
    Normalizedの場合visibilityが付属する
    '''
    landmark_dict = {}
    landmark_dict["x"] = landmark.x
    landmark_dict["y"] = landmark.y
    landmark_dict["z"] = landmark.z

    if hasattr(landmark, "visibility"):
        landmark_dict["visibility"] = landmark.visibility

    if hasattr(landmark, "presence"):
        landmark_dict["presence"] = landmark.presence

    return landmark_dict


def holistic_results_to_dict(results, json_dict={}, timestamp=None):
    '''holisticの結果をdictに変換する。
    他のデータを格納しているdictに追加するため引数でdictを取るようにする.

    @param[in] results holisticの計算結果
    @param[in] json_dict 渡された場合追加する (default: {})
    @param[in] timestamp ns単位、long int (default: None)
    '''

    # timestampeが指定されていない場合、この関数が呼ばれた瞬間
    if timestamp is None:
        timestamp = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    # 各landmarkへのアクセス方法
    # results.pose_landmarks.landmarkでiterableにアクセス可能

    # pose
    if hasattr(results, "pose_landmarks") and results.pose_landmarks is not None:
        json_dict["pose_landmarks"] = []
        for landmark in results.pose_landmarks:
            json_dict["pose_landmarks"].append(landmark_to_dict(landmark))
        json_dict["pose_landmarks_stamp"] = timestamp
    # pose3d
    if hasattr(results, "pose_world_landmarks") and results.pose_world_landmarks is not None:
        json_dict["pose_world_landmarks"] = []
        for landmark in results.pose_world_landmarks:
            json_dict["pose_world_landmarks"].append(landmark_to_dict(landmark))
        json_dict["pose_world_landmarks_stamp"] = timestamp
    # face
    if hasattr(results, "face_landmarks") and results.face_landmarks is not None:
        json_dict["face_landmarks"] = []
        for landmark in results.face_landmarks:
            json_dict["face_landmarks"].append(landmark_to_dict(landmark))
        json_dict["face_landmarks_stamp"] = timestamp
    # right hand
    if hasattr(results, "right_hand_landmarks") and results.right_hand_landmarks is not None:
        json_dict["right_hand_landmarks"] = []
        for landmark in results.right_hand_landmarks:
            json_dict["right_hand_landmarks"].append(landmark_to_dict(landmark))
        json_dict["right_hand_landmarks_stamp"] = timestamp
    # left hand
    if hasattr(results, "left_hand_landmarks") and results.left_hand_landmarks is not None:
        json_dict["left_hand_landmarks"] = []
        for landmark in results.left_hand_landmarks:
            json_dict["left_hand_landmarks"].append(landmark_to_dict(landmark))
        json_dict["left_hand_landmarks_stamp"] = timestamp

    return json_dict
