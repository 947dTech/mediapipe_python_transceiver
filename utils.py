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
        if landmark.visibility is not None:
            landmark_dict["visibility"] = landmark.visibility

    if hasattr(landmark, "presence"):
        if landmark.presence is not None:
            landmark_dict["presence"] = landmark.presence

    return landmark_dict


def category_to_dict(category):
    '''categoryをdictに変換する'''
    category_dict = {}
    category_dict[category.category_name] = category.score
    return category_dict


def holistic_results_to_dict(results, json_dict={}, timestamp=None,
                             legacy_face_mode=True,
                             face_blendshapes=False,
                             hand2d=True,
                             hand3d=False):
    '''holisticの結果をdictに変換する。
    他のデータを格納しているdictに追加するため引数でdictを取るようにする.

    @param[in] results holisticの計算結果
    @param[in] json_dict 渡された場合追加する (default: {})
    @param[in] timestamp ns単位、long int (default: None)
    @param[in] legacy_face_mode 新バージョンの478点->旧バージョンの468点に変換 (default: True)
    @param[in] hand2d {left,right}_hand_landmarksを含める (default: True)
    @param[in] hand3d {left,right}_hand_world_landmarksを含める (default: False)
    '''

    # timestampeが指定されていない場合、この関数が呼ばれた瞬間
    if timestamp is None:
        timestamp = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    # 各landmarkへのアクセス方法
    # results.pose_landmarksでiterableにアクセス可能

    # pose
    if (
        hasattr(results, "pose_landmarks") and
        results.pose_landmarks is not None
    ):
        json_dict["pose_landmarks"] = []
        for landmark in results.pose_landmarks:
            json_dict["pose_landmarks"].append(landmark_to_dict(landmark))
        json_dict["pose_landmarks_stamp"] = timestamp
    # pose3d
    if (
        hasattr(results, "pose_world_landmarks") and
        results.pose_world_landmarks is not None
    ):
        json_dict["pose_world_landmarks"] = []
        for landmark in results.pose_world_landmarks:
            json_dict["pose_world_landmarks"].append(landmark_to_dict(landmark))
        json_dict["pose_world_landmarks_stamp"] = timestamp
    # face
    # https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py
    if (
        hasattr(results, "face_landmarks") and
        results.face_landmarks is not None
    ):
        if legacy_face_mode and len(results.face_landmarks) == 478:
            # 前のバージョンでは468点だったが478点に増加している
            # 追加された10点はirisで単純に取り除くだけで互換性は取れる
            # 受信側で点数を見ているため取り除く
            face_landmarks = results.face_landmarks[:468]
        else:
            face_landmarks = results.face_landmarks
        json_dict["face_landmarks"] = []
        for landmark in face_landmarks:
            json_dict["face_landmarks"].append(landmark_to_dict(landmark))
        json_dict["face_landmarks_stamp"] = timestamp
    if (
        hasattr(results, "face_blendshapes") and
        results.face_blendshapes is not None and face_blendshapes
    ):
        json_dict["face_blendshapes"] = []
        for blendshape in results.face_blendshapes:
            json_dict["face_blendshapes"].append(category_to_dict(blendshape))
        # stampはface_landmarks_stampと同一になる
    # right hand
    if (
        hasattr(results, "right_hand_landmarks") and
        results.right_hand_landmarks is not None and hand2d
    ):
        json_dict["right_hand_landmarks"] = []
        for landmark in results.right_hand_landmarks:
            json_dict["right_hand_landmarks"].append(landmark_to_dict(landmark))
        json_dict["right_hand_landmarks_stamp"] = timestamp
    if (
        hasattr(results, "right_hand_world_landmarks") and
        results.right_hand_world_landmarks is not None and hand3d
    ):
        json_dict["right_hand_world_landmarks"] = []
        for landmark in results.right_hand_world_landmarks:
            json_dict["right_hand_world_landmarks"].append(landmark_to_dict(landmark))
        json_dict["right_hand_world_landmarks_stamp"] = timestamp
    # left hand
    if (
        hasattr(results, "left_hand_landmarks") and
        results.left_hand_landmarks is not None and hand2d
    ):
        json_dict["left_hand_landmarks"] = []
        for landmark in results.left_hand_landmarks:
            json_dict["left_hand_landmarks"].append(landmark_to_dict(landmark))
        json_dict["left_hand_landmarks_stamp"] = timestamp
    if (
        hasattr(results, "left_hand_world_landmarks") and
        results.left_hand_world_landmarks is not None and hand3d
    ):
        json_dict["left_hand_world_landmarks"] = []
        for landmark in results.left_hand_world_landmarks:
            json_dict["left_hand_world_landmarks"].append(landmark_to_dict(landmark))
        json_dict["left_hand_world_landmarks_stamp"] = timestamp

    return json_dict
