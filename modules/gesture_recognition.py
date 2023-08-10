import cv2
import mediapipe as mp
import math
from google.protobuf.json_format import MessageToDict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

'''
    求解二维向量角度
'''
def vector2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    
    try:
        angle = math.degrees(math.acos((v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle = 65535.
    
    if angle > 180.:
        angle = 65535.

    return angle

'''
    获取对应手的二维向量角度，根据角度确定手势
'''
def finger_angle(hand):
    angle_list = []

    # thumb 大拇指角度
    angle_list.append(vector2d_angle(
        ((int(hand[0][0]) - int(hand[2][0])), (int(hand[0][1]) - int(hand[2][1]))),
        ((int(hand[3][0]) - int(hand[4][0])), (int(hand[3][1]) - int(hand[4][1])))
    ))

    # index 食指角度
    angle_list.append(vector2d_angle(
        ((int(hand[0][0]) - int(hand[6][0])), (int(hand[0][1]) - int(hand[6][1]))),
        ((int(hand[7][0]) - int(hand[8][0])), (int(hand[7][1]) - int(hand[8][1])))
    ))

    # middle 中指角度
    angle_list.append(vector2d_angle(
        ((int(hand[0][0]) - int(hand[10][0])), (int(hand[0][1]) - int(hand[10][1]))),
        ((int(hand[11][0]) - int(hand[12][0])), (int(hand[11][1]) - int(hand[12][1])))
    ))

    # ring 无名指角度
    angle_list.append(vector2d_angle(
        ((int(hand[0][0]) - int(hand[14][0])), (int(hand[0][1]) - int(hand[14][1]))),
        ((int(hand[15][0]) - int(hand[16][0])), (int(hand[15][1]) - int(hand[16][1])))
    ))

    # pink 小拇指角度
    angle_list.append(vector2d_angle(
        ((int(hand[0][0]) - int(hand[18][0])), (int(hand[0][1]) - int(hand[18][1]))),
        ((int(hand[19][0]) - int(hand[20][0])), (int(hand[19][1]) - int(hand[20][1])))
    ))

    return angle_list

'''
    判断静态手势
'''
def gesture_judgment(angle_list):
    thr_angle = 65.
    thr_angle_thumb = 53.
    thr_angle_s = 49.

    gesture_str = "none"
    if 65535. not in angle_list:
        if angle_list[0] > thr_angle_thumb and angle_list[1] > thr_angle and angle_list[2] > thr_angle and angle_list[3] > thr_angle and angle_list[4] > thr_angle:
            gesture_str = "拳头"
        elif angle_list[0] < thr_angle_thumb and angle_list[1] > thr_angle and angle_list[2] > thr_angle and angle_list[3] > thr_angle and angle_list[4] > thr_angle:
            gesture_str = "大拇指"
        elif angle_list[0] > thr_angle_thumb and angle_list[1] < thr_angle and angle_list[2] > thr_angle and angle_list[3] > thr_angle and angle_list[4] > thr_angle:
            gesture_str = "一/食指"
        elif angle_list[0] > thr_angle_thumb and angle_list[1] > thr_angle and angle_list[2] < thr_angle and angle_list[3] > thr_angle and angle_list[4] > thr_angle:
            gesture_str = "鄙视/中指"
        elif angle_list[0] > thr_angle_thumb and angle_list[1] > thr_angle and angle_list[2] > thr_angle and angle_list[3] < thr_angle and angle_list[4] > thr_angle:
            gesture_str = "无名指"
        elif angle_list[0] > thr_angle_thumb and angle_list[1] > thr_angle and angle_list[2] > thr_angle and angle_list[3] > thr_angle and angle_list[4] < thr_angle:
            gesture_str = "小拇指"
        elif angle_list[0] > thr_angle_thumb and angle_list[1] < thr_angle and angle_list[2] < thr_angle and angle_list[3] > thr_angle and angle_list[4] > thr_angle:
            gesture_str = "剪刀/二"
        elif angle_list[0] > thr_angle_thumb and angle_list[1] < thr_angle and angle_list[2] < thr_angle and angle_list[3] < thr_angle and angle_list[4] > thr_angle:
            gesture_str = "三"
        elif angle_list[0] > thr_angle_thumb and angle_list[1] < thr_angle and angle_list[2] < thr_angle and angle_list[3] < thr_angle and angle_list[4] < thr_angle:
            gesture_str = "四"
        elif angle_list[0] < thr_angle_thumb and angle_list[1] < thr_angle and angle_list[2] < thr_angle and angle_list[3] < thr_angle and angle_list[4] < thr_angle:
            gesture_str = "五"
        elif angle_list[0] < thr_angle_thumb and angle_list[1] > thr_angle and angle_list[2] > thr_angle and angle_list[3] > thr_angle and angle_list[4] < thr_angle:
            gesture_str = "六"
        elif angle_list[0] < thr_angle_thumb and angle_list[1] < thr_angle and angle_list[2] > thr_angle and angle_list[3] > thr_angle and angle_list[4] > thr_angle:
            gesture_str = "七"
    return gesture_str

'''
    动态手势变化计算
'''
def dynamic_point(dynamic_list):
    if len(dynamic_list) > 20:
        del(dynamic_list[0])

    x = dynamic_list[0][0] - dynamic_list[len(dynamic_list) - 1][0]
    y = dynamic_list[0][1] - dynamic_list[len(dynamic_list) - 1][1]

    return (x, y)

'''
    动态手势判断
'''
def dynamic_gesture(line):
    dynamic_gesture_str = ""
    if line[0] > 20:
        dynamic_gesture_str += "/向左"
    elif line[0] < -20:
        dynamic_gesture_str += "/向右"

    if line[1] > 20:
        dynamic_gesture_str += "/向上"
    elif line[1] < -20:
        dynamic_gesture_str += "/向下"

    return dynamic_gesture_str



'''
    cv2中文显示
'''
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def start_module():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.82,
            min_tracking_confidence=0.82)
    cap = cv2.VideoCapture(0)

    p_time = 0
    c_time = 0
    
    dynamic_list_l= []
    dynamic_list_r= []

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            if len(results.multi_handedness) == 2:
                frame =  cv2AddChineseText(
                    frame,
                    "检测到双手",
                    (260, 25),
                    (0, 255, 0),
                    20)                            
            else:
                frame = cv2AddChineseText(
                    frame,
                    "检测到单手",
                    (260, 25),
                    (0, 255, 0),
                    20)
            
            f_num = 0
            label_list = []
            for handedness in results.multi_handedness:
                label_list.append(MessageToDict(handedness)['classification'][0]['label'])

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_local = []


                for i in range(21):
                    x = hand_landmarks.landmark[i].x * frame.shape[1]
                    y = hand_landmarks.landmark[i].y * frame.shape[0]
                    hand_local.append((x, y))

                if hand_local:
                    angle_list = finger_angle(hand_local)
                    gesture_str = gesture_judgment(angle_list)
                    if label_list[f_num] == "Left":
                        frame = cv2AddChineseText(
                            frame,
                            "左手静态手势:" + gesture_str,
                            (10, 100),
                            (0, 255, 0),
                            20)
                        dynamic_list_l.append(hand_local[8])
                        dynamic_gesture_str = dynamic_gesture(dynamic_point(dynamic_list_l))

                        frame = cv2AddChineseText(
                            frame,
                            "左手动态手势:" + dynamic_gesture_str,
                            (10, 200),
                            (0, 255, 0),
                            20)

                    else:
                        frame = cv2AddChineseText(
                            frame,
                            "右手静态手势:" + gesture_str,
                            (400, 100),
                            (0, 255, 0),
                            20)
                        dynamic_list_r.append(hand_local[8])
                        dynamic_gesture_str = dynamic_gesture(dynamic_point(dynamic_list_r))

                        frame = cv2AddChineseText(
                            frame,
                            "右手动态手势:" + dynamic_gesture_str,
                            (400, 200),
                            (0, 255, 0),
                            20)

                    # print(str(dynamic_num_x) + "," + str(dynamic_num_y))


                f_num += 1
        
        c_time = time.time()
        fps = int(1 / (c_time - p_time))
        p_time = c_time

        frame = cv2AddChineseText(
            frame,
            "fps:" + str(fps),
            (10, 10),
            (0, 255, 0),
            20)


        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
if __name__ == '__main__':
    start_module()
