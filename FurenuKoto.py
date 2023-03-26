import cv2
import pyaudio
import numpy as np
import mediapipe as mp
import threading
from queue import Queue
from google.protobuf.json_format import MessageToDict


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


# 音声関連のパラメータ
sampling_rate = 44100
duration = 0.07
frequency = 440  # 基本周波数
start_pos = 0  # 音がブツブツなるのを防ぐ
end_pos = 0

# PyAudioのインスタンスを作成する
p = pyaudio.PyAudio()

# Streamを開く
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=sampling_rate,
                output=True)

# 音声を生成する
def generate_sound(frequency, queue):
    global start_pos
    global end_pos
    print('start => end', start_pos, end_pos)
    start_pos = end_pos
    end_pos = start_pos + duration * sampling_rate
    time = np.arange(start_pos, end_pos) / sampling_rate
    wave = np.sin(2 * np.pi * frequency * time)
    queue.put(wave)

# ストリームで音声を再生する
def play_sound(stream, queue):
    while True:
        wave = queue.get()
        stream.write(wave.astype(np.float32).tobytes())

# スレッドで実行する関数
# メインスレッドで実行する部分
cap = cv2.VideoCapture(0)
queue = Queue()

# スレッドで波形生成を実行する
thread_generate_sound = threading.Thread(target=generate_sound, args=(frequency, queue))
thread_generate_sound.start()

# スレッドで音声再生を実行する
thread_play_sound = threading.Thread(target=play_sound, args=(stream, queue))
thread_play_sound.start()

# スケールが変更されたか
isChanged_scale = False
m_scale = 0

# 画面分割用の計算
_, im = cap.read()
threshold_x = [im.shape[1] * i // 13 for i in range(13)]
threshold_y = [im.shape[0] * i // 3 for i in range(3)]
threshold_y_blackkey = [0, im.shape[0] * 1 // 4, im.shape[0] * 3 // 4]  #  黒鍵は短し

# 画面のタイル作成用
rows = 3
columns = 13
rect_width = int(im.shape[1] // columns)
rect_height = int(im.shape[0] // rows)
black_key = [2, 4, 7, 9, 11]


while True:

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    
        while True:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
    
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            # 手のx, y座標を、mediapipeを用いて、取り出す
            x = 0
            y = 0
            hand_landmarks_dict = [0] * 3
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                    # 手のひらの座標を検出
                    hand_landmarks_dict[idx] = MessageToDict(hand_landmarks)
                    x = int(hand_landmarks_dict[idx]['landmark'][9]['x'] * im.shape[1])
                    y = int(hand_landmarks_dict[idx]['landmark'][9]['y'] * im.shape[0])
                    #cv2.drawMarker(image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=30)
                    cv2.circle(image, (x, y), 13, (31, 113, 237), thickness=-1)


                    # 手の位置から、音階を求める
                    scale = 0
                    for idx, i in enumerate(threshold_x):
                        if x > i:
                            scale = idx
                    
                    # 手の位置から、オクターブを求める
                    # oc = 1の時は、音を鳴らさない
                    oc = 0
                    for idx, i in enumerate(threshold_y):
                        if scale+1 in black_key:
                            if y > threshold_y_blackkey[idx]:
                                oc = idx
                        else:
                            if y > i:
                                oc = idx
                    
                    if oc != 1:
                        # 音階が変化したことを感知する
                        if m_scale != scale:
                            isChanged_scale = True
                            m_scale = scale

                        # 音階から、周波数を求める
                        f = 523.251
                        if oc == 2:
                            f = 261.626
                        if scale != 0:
                            for i in range(scale):
                                f *= 1.059463094

                        if isChanged_scale:
                            frequency = f
                            thread_generate_sound.join()
                            thread_generate_sound = threading.Thread(target=generate_sound, args=(frequency, queue))
                            thread_generate_sound.start()

                            print(scale)
                            isChanged_scale = False
                    else:
                        isChanged_scale = True

            if cv2.waitKey(5) & 0xFF == 27:
                break

            # 画面にタイルを貼る
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            tile_color = 0
            p1 = 0
            p2 = 0
            tile_image = image.copy()  # タイルを表示させるためのimage
            for i in range(rows):
                for k in range(columns):
                    if i != 1:

                        # 黒鍵の描き分け
                        if k+1 in black_key:
                            tile_color = (0, 0, 0)
                            if i == 0:
                                p1 = (k * rect_width + 5, i * rect_height + 5)
                                p2 = ((k+1) * rect_width - 5, threshold_y_blackkey[1])
                            else:
                                p1 = (k * rect_width + 5, threshold_y_blackkey[2])
                                p2 = ((k+1) * rect_width - 5, (i+1) * rect_height - 5)

                        else:
                            p1 = (k * rect_width + 5, i * rect_height + 5)
                            p2 = ((k+1) * rect_width - 5, (i+1) * rect_height - 5)
                            tile_color = (255, 255, 255)


                        cv2.rectangle(tile_image, p1, p2, tile_color, thickness=-1)

            # タイルを半透明にして、カメラ画像と結合
            mat_image = cv2.addWeighted(tile_image, 0.4, image, 0.6, 0)
            cv2.imshow('MediaPipe Hands', mat_image)

# ストリームを閉じる
stream.stop_stream()
stream.close()

# PyAudioを終了する
p.terminate()

cv2.destroyAllWindows()
