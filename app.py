from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None
draw_color = (255, 0, 255)
brush_thickness = 10
eraser_thickness = 50
xp, yp = 0, 0

def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def gen_frames():
    global canvas, xp, yp, draw_color

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            if canvas is None:
                canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((cx, cy))

                    fingers = fingers_up(hand_landmarks)

                    if fingers[1] and fingers[2]:
                        xp, yp = 0, 0
                        if lm_list[8][1] < 100:
                            draw_color = (255, 0, 255)
                        elif lm_list[8][1] < 200:
                            draw_color = (255, 0, 0)
                        elif lm_list[8][1] < 300:
                            draw_color = (0, 255, 0)
                        elif lm_list[8][1] < 400:
                            draw_color = (0, 0, 0)
                    elif fingers[1] and not fingers[2]:
                        cx, cy = lm_list[8]
                        if xp == 0 and yp == 0:
                            xp, yp = cx, cy
                        if draw_color == (0, 0, 0):
                            cv2.line(frame, (xp, yp), (cx, cy), draw_color, eraser_thickness)
                            cv2.line(canvas, (xp, yp), (cx, cy), draw_color, eraser_thickness)
                        else:
                            cv2.line(frame, (xp, yp), (cx, cy), draw_color, brush_thickness)
                            cv2.line(canvas, (xp, yp), (cx, cy), draw_color, brush_thickness)
                        xp, yp = cx, cy

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, inv_canvas = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
            inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, inv_canvas)
            frame = cv2.bitwise_or(frame, canvas)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
