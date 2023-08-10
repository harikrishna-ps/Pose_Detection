import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import joblib

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

model = load_model('YogaPoseModel.h5')
# Load the preprocessing pipeline and label encoder
preprocessing_pipeline = joblib.load('preprocessing_pipeline.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Set a confidence threshold
confidence_threshold = 0.70


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

    

# Curl counter variables
counter = 0 
stage = ""
count_inc = False


option = st.selectbox('Select Mode',
    ("Hands Up", "Wide Hands", "Yoga Pose"))


# Setup mediapipe instance
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose =  mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def frame_processing(frame):
        
    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Make detection
    results = pose.process(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark
    
        # Get left arm coordinates
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        
            # Get right arm coordinates
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        # Calculate angles for left arm
        angle1_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
        angle2_left = calculate_angle(left_hip, left_shoulder, left_elbow)
        
        # Calculate angles for right arm
        angle1_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
        angle2_right = calculate_angle(right_hip, right_shoulder, right_elbow)
        
        # cv2.putText(image, f"{angle1_left:.2f}", 
        #             tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(image, f"{angle1_right:.2f}", 
        #             tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # # Additional lines to show angle2
        # cv2.putText(image, f"{angle2_left:.2f}", 
        #             tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(image, f"{angle2_right:.2f}", 
        #             tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if option=="Hands Up":
            if angle1_left > 160 and 160 < angle2_left < 190 and angle1_right > 160 and 160 < angle2_right < 190:
                stage = True
            else:
                stage = False
            if stage==True:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                        )
            else:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                        )
        
        elif option=="Wide Hands":
            if angle1_left > 155 and 80 < angle2_left < 110 and angle1_right > 155 and 80 < angle2_right < 110:
                stage = True
            else:
                stage = False
            if stage==True:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                        )
            else:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                        )
            
        elif option=="Yoga Pose":
            image_data = {}
            for i in range(33):
                image_data[f'{mp_pose.PoseLandmark(i).name}-x'] = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x
                image_data[f'{mp_pose.PoseLandmark(i).name}-y'] = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y
                image_data[f'{mp_pose.PoseLandmark(i).name}-z'] = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z
                image_data[f'{mp_pose.PoseLandmark(i).name}-visibility'] = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility

            test_df = pd.DataFrame([image_data])
            predictions_encoded = model.predict(test_df)
            predictions = label_encoder.inverse_transform(predictions_encoded.argmax(axis=1))
            for pred, conf_score in zip(predictions, predictions_encoded.max(axis=1)):
                if conf_score >= confidence_threshold:
                    cv2.putText(image, f"Prediction: {pred}", (30,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                        )
                else:
                    cv2.putText(image, f"Prediction: Unknown", (30,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 66, 230), 2, cv2.LINE_AA)
                    
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                        )
            
    return(image)

def callback(frame):
    frame = frame.to_ndarray(format="bgr24")
    image = frame_processing(frame)
    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(key="Pose Estimation", mode=WebRtcMode.SENDRECV, 
                rtc_configuration=RTC_CONFIGURATION,
                video_frame_callback=callback, 
                media_stream_constraints={"video": True, "audio": False}, 
                async_processing=True)
    
