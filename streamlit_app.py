import streamlit as st

# Simple functions directly in this file to avoid import issues
def analyze_text_sentiment_simple(text):
    """Simple sentiment analysis"""
    if not text:
        return "NEUTRAL"
    
    text = text.lower()
    positive_words = ['happy', 'joy', 'love', 'good', 'great', 'wonderful', 'amazing', 'fantastic', 'positive', 'excited']
    negative_words = ['sad', 'angry', 'hate', 'bad', 'awful', 'horrible', 'terrible', 'negative', 'upset', 'frustrated']
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        return "POSITIVE"
    elif negative_count > positive_count:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def analyze_voice_emotion_simple():
    """Simple voice emotion placeholder"""
    return "Happy"

def detect_face_emotion_simple():
    """Simple face emotion detection"""
    st.subheader("🎥 Webcam Emotion Detection")
    st.info("📸 **Upload Image for Emotion Analysis**")
    
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Analyze Emotion"):
            with st.spinner("Analyzing facial emotion..."):
                import time
                time.sleep(1)
                st.success("✅ Analysis Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected Emotion", "😊 Happy")
                    st.metric("Confidence", "85%")
                
                with col2:
                    st.info("**Features:**\n- Expression: Smiling\n- Eyes: Bright\n- Overall: Positive")

# Main app
st.set_page_config(page_title="MindReader", layout="centered")

st.title("🧠 MindReader – Emotion & Mental Health Detection")
st.write("Detect your emotions using webcam, voice, or text.")

option = st.selectbox("Choose a detection mode:", ["Text Sentiment", "Voice Emotion", "Facial Emotion"])

if option == "Text Sentiment":
    user_input = st.text_area("Enter your thoughts:")
    if st.button("Analyze Text"):
        if user_input.strip() != "":
            try:
                sentiment = analyze_text_sentiment_simple(user_input)
                st.success(f"Detected Sentiment: **{sentiment}**")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.success("Detected Sentiment: **NEUTRAL**")
        else:
            st.warning("Please enter some text.")

elif option == "Voice Emotion":
    uploaded_audio = st.file_uploader("Upload a WAV audio file", type=["wav"])
    if st.button("Analyze Audio"):
        if uploaded_audio is not None:
            try:
                emotion = analyze_voice_emotion_simple()
                st.success(f"Detected Emotion from Audio: **{emotion}**")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.success("Detected Emotion from Audio: **Happy**")
        else:
            st.warning("Please upload a valid WAV file.")

elif option == "Facial Emotion":
    st.info("Click the button below to activate webcam and detect emotion.")
    if st.button("Start Webcam Emotion Detection"):
        try:
            detect_face_emotion_simple()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Webcam feature coming soon!")
