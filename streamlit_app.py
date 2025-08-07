import streamlit as st
from app.webcam_emotion import detect_face_emotion
from app.voice_emotion import analyze_voice_emotion
from app.text_sentiment import analyze_text_sentiment

st.set_page_config(page_title="MindReader", layout="centered")

st.title("🧠 MindReader – Emotion & Mental Health Detection")
st.write("Detect your emotions using webcam, voice, or text.")

option = st.selectbox("Choose a detection mode:", ["Text Sentiment", "Voice Emotion", "Facial Emotion"])

if option == "Text Sentiment":
    user_input = st.text_area("Enter your thoughts:")
    if st.button("Analyze Text"):
        if user_input.strip() != "":
            sentiment = analyze_text_sentiment(user_input)
            st.success(f"Detected Sentiment: **{sentiment}**")
        else:
            st.warning("Please enter some text.")

elif option == "Voice Emotion":
    uploaded_audio = st.file_uploader("Upload a WAV audio file", type=["wav"])
    if st.button("Analyze Audio"):
        if uploaded_audio is not None:
            emotion = analyze_voice_emotion(uploaded_audio)
            st.success(f"Detected Emotion from Audio: **{emotion}**")
        else:
            st.warning("Please upload a valid WAV file.")

elif option == "Facial Emotion":
    st.info("Click the button below to activate webcam and detect emotion.")
    if st.button("Start Webcam Emotion Detection"):
        detect_face_emotion()
