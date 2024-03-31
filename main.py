import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import streamlit as st
import pandas as pd
from googletrans import Translator
from gtts import gTTS
import pygame
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

model_path = "my_model.h5"
loaded_model = load_model(model_path)

class_labels = [
    "acanthosis_nigricans",
    "acne",
    "bullous_pemphigoid",
    "candidiasis_mouth",
    "dermagraphism",
    "eczema",
    "impetigo",
    "lupus_chronic_cutaneous",
    "molluscum_contagiosum",
    "nevus"
]

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    predictions = loaded_model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class

st.set_page_config(layout="wide")

st.sidebar.title("Mouse Control Options")
control_method = st.sidebar.radio("Select Control Method", ("Hand Control", "Eye Control", "Manual Control"))

cap = None  

if control_method == "Hand Control":
    st.sidebar.subheader("Hand-Controlled Mouse")
    st.sidebar.write("To use hand-controlled mouse, run the hand-controlled code.")

    if st.sidebar.button("Run Hand-Controlled Code"):
        cap = cv2.VideoCapture(0)
        hand_detector = mp.solutions.hands.Hands()
        drawing_utils = mp.solutions.drawing_utils
        screen_width, screen_height = pyautogui.size()
        index_y = 0

        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = hand_detector.process(rgb_frame)
            hands = output.multi_hand_landmarks
            if hands:
                for hand in hands:
                    drawing_utils.draw_landmarks(frame, hand)
                    landmarks = hand.landmark
                    for id, landmark in enumerate(landmarks):
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        if id == 8:
                            cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                            index_x = screen_width / frame_width * x
                            index_y = screen_height / frame_height * y
                            pyautogui.moveTo(index_x, index_y)
                        if id == 4:
                            cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                            thumb_x = screen_width / frame_width * x
                            thumb_y = screen_height / frame_height * y
                            if abs(index_y - thumb_y) < 50:
                                pyautogui.click()
                                pyautogui.sleep(1)
        cv2.imshow('Virtual Mouse', frame)
        cv2.waitKey(1)

elif control_method == "Eye Control":
    st.sidebar.subheader("Eye-Controlled Mouse")
    st.sidebar.write("To use eye-controlled mouse, run the eye-controlled code.")

    if st.sidebar.button("Run Eye-Controlled Code"):
        cam = cv2.VideoCapture(0)
        face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        screen_w, screen_h = pyautogui.size()

        while True:
            _, frame = cam.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = face_mesh.process(rgb_frame)
            landmark_points = output.multi_face_landmarks
            frame_h, frame_w, _ = frame.shape

            if landmark_points:
                landmarks = landmark_points[0].landmark
                for id, landmark in enumerate(landmarks[474:478]):
                    x = int(landmark.x * frame_w)
                    y = int(landmark.y * frame_h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0))
                    if id == 1:
                        screen_x = screen_w / frame_w * x
                        screen_y = screen_h / frame_h * y
                        pyautogui.moveTo(screen_x, screen_y)
                    left = [landmarks[145], landmarks[159]]
                    for landmark in left:
                        x = int(landmark.x * frame_w)
                        y = int(landmark.y * frame_h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 255))
                    if (left[0].y - left[1].y) < 0.008:
                        pyautogui.click()
                        pyautogui.sleep(1)
        cv2.imshow('Eye Controlled Mouse', frame)
        cv2.waitKey(1)

elif control_method == "Manual Control":
    st.sidebar.subheader("Manual Mouse Control")
    st.sidebar.write("You can use manual mouse control.")

st.title("Skin Cam and Disease Classifier")

uploaded_image = st.file_uploader("Upload an image for disease prediction", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:

    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    predicted_disease = predict_disease(uploaded_image)
    st.write(f"Predicted Disease: {predicted_disease}")

    @st.cache
    def load_data():
        df = pd.read_excel("C:\\Users\\pradeep\\Desktop\\skin new.xlsx", engine='openpyxl')
        return df

    def translate_text(text, target_language):
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text

    def save_speech_to_file(text, file_name, language='en'):
        speech = gTTS(text=text, lang=language, slow=False)
        speech.save(file_name)

    def play_audio_from_file(file_name):
        pygame.mixer.init()
        pygame.mixer.music.load(file_name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def delete_audio_file(file_name):
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            os.remove(file_name)
        except FileNotFoundError:
            pass

    st.title("Disease Information Lookup")

    disease_data = load_data()

    disease_name = predicted_disease

    target_language = st.selectbox("Select Target Language", ["te", "fr", "es", "de", "ru", "ta", "ur", "ml", "bn", "hi", "kn"])

    disease_info = disease_data[disease_data['Disease'] == disease_name]

    if not disease_info.empty:
        st.subheader('Disease Information')
        st.write(f'Disease Name: {disease_info.iloc[0]["Disease"]}')
        st.write(f'Description: {disease_info.iloc[0]["Description"]}')
        st.write(f'Diagnosis: {disease_info.iloc[0]["Diagnosis"]}')
        st.write(f'Medication: {disease_info.iloc[0]["Medication"]}')

        translated_description = translate_text(disease_info.iloc[0]["Description"], target_language)
        translated_diagnosis = translate_text(disease_info.iloc[0]["Diagnosis"], target_language)
        translated_medication = translate_text(disease_info.iloc[0]["Medication"], target_language)

        st.header(f"Translated Information (to {target_language})")
        st.write(f'Description (Translated): {translated_description}')
        st.write(f'Diagnosis (Translated): {translated_diagnosis}')
        st.write(f'Medication (Translated): {translated_medication}')

        info_to_speak = f"{disease_info.iloc[0]['Description']}. {disease_info.iloc[0]['Diagnosis']}. {disease_info.iloc[0]['Medication']}"
        audio_file_name = "information.mp3"
        save_speech_to_file(info_to_speak, audio_file_name, 'en')

        if st.button("Play Information (English)"):
            play_audio_from_file(audio_file_name)
