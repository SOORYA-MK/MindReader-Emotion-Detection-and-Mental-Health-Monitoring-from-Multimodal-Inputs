
import streamlit as st
from app.webcam_emotion_enhanced import detect_face_emotion
from app.voice_emotion import analyze_voice_emotion
from app.text_sentiment import analyze_text_sentiment
from app.mood_tracker import track_mood
from app.model_accuracy import show_model_accuracy
from app.dataset_info import show_dataset_info

st.title("🧠 MindReader - Emotion & Mental Health Detector")

menu = st.sidebar.selectbox("Choose Input Mode", [
    "🎥 Enhanced Webcam", 
    "Voice", 
    "Text", 
    "Mood Tracker", 
    "📊 Model Accuracy",
    "🗃️ Dataset Info"
])

if menu == "🎥 Enhanced Webcam":
    st.info("Enhanced AI Facial Emotion Detection with Post-processing")
    detect_face_emotion()

elif menu == "Voice":
    st.info("Upload Voice File or Record")
    analyze_voice_emotion()

elif menu == "Text":
    st.info("Text Sentiment & Emotion Analysis")
    analyze_text_sentiment()

elif menu == "Mood Tracker":
    st.info("Your Mood Timeline")
    track_mood()

elif menu == "📊 Model Accuracy":
    show_model_accuracy()

elif menu == "🗃️ Dataset Info":
    show_dataset_info()
