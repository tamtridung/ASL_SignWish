import streamlit as st
import cv2
import numpy as np
import os
import time
import mediapipe as mp
import pyttsx3
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# MP Holistic:
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

# Extract Keypoint values
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# Load model:
model = tf.keras.models.load_model('./pretrained_models/model_11classes_v3.h5')

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Visualize prediction:
def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()

    pred_dict = dict(zip(actions, res))
    # sorting for prediction and get top 5
    prediction = sorted(pred_dict.items(), key=lambda x: x[1])[::-1][:5]

    for num, pred in enumerate(prediction):
        text = '{}: {}'.format(pred[0], round(float(pred[1]),4))
        # cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, text, (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA) 
    return output_frame

# New detection variables
sequence = []
sentence = []
threshold = 0.9
tts = False
actions = os.listdir('./MP_Data')
label_map = {label:num for num, label in enumerate(actions)}

# Text to speak config:
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[10].id)

###############################################################################################
                                            # STREAMLIT #

col1, col2 = st.columns((3,1))
with col1:
    st.title('SIGN WISH')
    st.write('Make by TAM TRAN')

with col2:
    st.image('./streamlit_files/asl-icon.png')

# Checkboxes
st.header('Webcam')

col1, col2, col3 = st.columns(3)
with col1:
    show_webcam = st.checkbox('Show webcam')
with col2:
    show_landmarks = st.checkbox('Show landmarks')
with col3:
    speak = st.checkbox('Speak')

# Webcam
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0) # device 1/2

# Mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while show_webcam:
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        if show_landmarks:
            draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)

        sequence.append(keypoints)
        sequence = sequence[-24:]
        
        if len(sequence) == 24:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            #3. Viz logic
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        # incase the first word is iloveyou:
                        if (sentence[0] == '') and (actions[np.argmax(res)] == 'i love you'):
                            pass
                        else:
                            sentence.append(actions[np.argmax(res)])
                            tts = True
                else:
                    sentence.append(actions[np.argmax(res)])
                    tts = True

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            if show_landmarks:
                image = prob_viz(res, actions, image)

            # Text to speak:
            if speak:
                if tts: 
                    engine.say(sentence[-1])
                    engine.runAndWait()
                    # time.sleep(0.5)
                    tts = False

            # show result
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        # cv2.imshow('OpenCV Feed', image)
        frameshow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frameshow)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
