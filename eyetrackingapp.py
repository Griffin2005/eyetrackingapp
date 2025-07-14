import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math

st.set_page_config(page_title="ADAPTIFY Eye Tracking", layout="wide")
st.title("👁️ ADAPTIFY - Eye Tracking Demo")

# Initialize mediapipe face mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

FRAME_WINDOW = st.image([])

start_demo = st.button("Start Eye Tracking")

screen_w, screen_h = 1280, 720  # Simulated screen size for accuracy %

def calculate_accuracy(x, y, target_x=640, target_y=360):
    error_distance = math.sqrt((target_x - x) ** 2 + (target_y - y) ** 2)
    screen_diagonal = math.sqrt(screen_w ** 2 + screen_h ** 2)
    movement_accuracy = 100 - (error_distance / screen_diagonal) * 100
    return max(min(movement_accuracy, 100), 0)

if start_demo:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam access failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        frame_h, frame_w, _ = frame.shape
        simulated_mouse_x, simulated_mouse_y = None, None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            for id in [474, 475, 476, 477]:
                x = int(landmarks[id].x * frame_w)
                y = int(landmarks[id].y * frame_h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                if id == 475:
                    simulated_mouse_x = x
                    simulated_mouse_y = y

            # Simulate mouse indicator
            if simulated_mouse_x and simulated_mouse_y:
                cv2.circle(frame, (simulated_mouse_x, simulated_mouse_y), 20, (0, 0, 255), 2)
                accuracy = calculate_accuracy(simulated_mouse_x, simulated_mouse_y)
                cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Blink detection visualization (optional, no click)
            left_eye = [landmarks[145], landmarks[159]]
            eye_distance = left_eye[0].y - left_eye[1].y
            for landmark in left_eye:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
            if eye_distance < 0.01:
                cv2.putText(frame, 'Blink Detected (Simulated Click)', (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
