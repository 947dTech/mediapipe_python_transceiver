# mediapipe_python_transceiver

- 著者: 矢口 裕明 (クシナダ機巧株式会社)
- Author: Hiroaki Yaguchi (947D-tech)

## 本ソフトウェアについて

mediapipeのpythonバインディングを用いたサンプルプログラムです。
webカメラからの映像に対してholistic trackingを適用し、その結果をUDP送信します。
結果はHIROMEIROやmediapipe_receiver_utilityで受信することができます。

Apache2.0ライセンスです。
一部にmediapipeのサンプルコードを利用しています。

https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/docs/solutions/holistic.md

## 動作確認環境

- OS: Ubuntu 24.04
- Webカメラ: Logicool C920
- python環境構築にuvを使用しています

## 使用方法

### インストール

本リポジトリはuvで管理されています。
uvでprojectをインポートしてください。

libopencv-devは事前にapt等でインストールを行ってください。

### webカメラの設定を確認

UVC形式のwebカメラを接続し、開けることを事前に確認してください。
また、カメラで設定できる解像度やフレームレートなどは事前に確認してください。

v4l-utilsがインストールされている場合、以下のように確認できます。

```bash
$ v4l2-ctl --list-formats-ext
```

### カメラキャリブレーション

正確な結果を得るためにキャリブレーションを行ってください。
opencvのappに含まれるopencv_interactive-calibrationを使用します。
このアプリケーションはaptに含まれないため、
例えば、以下のようにビルドを行います。

```bash
$ git clone https://github.com/opencv/opencv.git
$ cd opencv/
$ git checkout 4.12.0
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_apps=ON ..
$ make opencv_interactive-calibration -j8
```

これでbinの下にopencv_interactive-calibrationが生成されます。

使用方法は公式サイトを参照してください。
ただし、これだけでは分かりづらい箇所があるため、以下で説明を行います。

https://docs.opencv.org/4.x/d7/d21/tutorial_interactive_calibration.html

まず、キャリブレーションボードを用意してください。
キャリブレーションボードは例えば以下のサイトで作成できます。
Asymmetric Circlesを選択すると、アプリ側のcirclesに対応します。

https://calib.io/pages/camera-calibration-pattern-generator

PDFを印刷するか、ディスプレイに表示しても実行できます。
サイズが重要なので必ず100%に拡大して下さい。

次にdefaultConfig.xmlをマニュアルを参照してアプリと同じ場所に作成してください。
この中で、解像度を指定する項目があるので、適宜変えてください。
実行時のオプションには解像度指定項目がないため、このファイルで設定してください。

実行は例えば以下のようにします。
sz,w,hは出力したボードに合わせてください。

```bash
~/opencv/build/bin$ ./opencv_interactive-calibration -t=circles -sz=15 -w=4 -h=11
```

10-20枚程度姿勢を変えて撮影したあとsを押すとcameraParameters.xmlが出力されます。
これをリポジトリ直下にコピーしてください。

### 実行

uvの環境が設定されている状態で、以下のコマンドを実行してください。

```bash
$ python3 holistic_transceiver.py
```

以下のオプションが利用できます。

- `-i|--input`: 入力デバイスID、default=0
- `-d|--device`: 入力デバイスファイル、inputとdeviceはdeviceが優先されます。
- `-r|--rate`: フレームレート、default=30
- `--width`: 横解像度、default=1920
- `--height`: 縦解像度、default=1080
- `--codec`: コーデックを4文字で、default=MJPG
- `-c|--calib_file`: キャリブレーションファイル、default=cameraParameters.xml
- `--host`: 送信先ホスト名、default=192.168.11.19
- `--port`: ポート番号(default=0x947d,変更する必要は通常ありません)
- `--msg-mode`: 送信メッセージの型を指定します。default=legacy
    - legacy: faceを468点、handを2Dで送信します。
    - latest: faceを478点、face blendshapesを含め、handを3Dで送信します。
    - full: すべての内容を送信します。
- `--hide-view`: 指定すると認識結果を表示しません。顔バレ防止と速度向上になります。何が認識されているかはわかりにくくなります。


## 送信データ仕様

ポート番号0x947dに対してUDPで送信します。
モードにより送信される内容が異なります。

- camera_params
    - focal_length: キャリブレーションで求めた焦点距離(単位:ピクセル)
    - cx: 画像中心のx座標(単位:ピクセル)
    - cy: 画像中心のy座標(単位:ピクセル)
    - frame_width: 横解像度
    - frame_height: 縦解像度
- gravity: 重力方向のベクトルの三次元配列
- gravity_stamp: タイムスタンプ(単位:ns)
- pose_landmarks: 二次元空間内の姿勢推定結果
- pose_landmarks_stamp: タイムスタンプ(単位:ns)
- pose_world_landmarks: 三次元空間内の姿勢推定結果
- pose_world_landmarks_stamp: タイムスタンプ(単位:ns)
- face_landmarks: 二次元空間内の顔の姿勢推定結果
- face_blendshapes: 表情パラメーター、52個を名前:値で格納
- face_landmarks_stamp: タイムスタンプ(単位:ns)
- right_hand_landmarks: 二次元空間内の右手の姿勢推定結果
- right_hand_landmarks_stamp: タイムスタンプ(単位:ns)
- right_hand_world_landmarks: 三次元空間内の右手の姿勢推定結果
- right_hand_world_landmarks_stamp: タイムスタンプ(単位:ns)
- left_hand_landmarks: 二次元空間内の左手の姿勢推定結果
- left_hand_landmarks_stamp: タイムスタンプ(単位:ns)
- left_hand_world_landmarks: 三次元空間内の左手の姿勢推定結果
- left_hand_world_landmarks_stamp: タイムスタンプ(単位:ns)

タイムスタンプはすべて同じ値=画像が取得された時刻を入れています。

### 重力ベクトルの定義について

Androidに準拠した座標系になります。画像座標系ではありません。

https://developer.android.com/develop/sensors-and-location/sensors/sensors_overview?hl=ja#sensors-coords

Y-up, X-left, Z-frontです。

まっすぐカメラを立てた場合、`(0.0, 9.8, 0.0)`となります。
