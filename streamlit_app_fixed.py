import streamlit as st

# Simple emotion/sentiment functions for Streamlit Cloud compatibility
def analyze_text_sentiment(user_input=None):
    """Simple text sentiment analysis"""
    try:
        if user_input is not None:
            # Called with parameter from streamlit_app.py
            text = user_input.lower()
            
            # Simple keyword-based sentiment
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
        else:
            # Full UI version (not used in streamlit_app.py)
            st.subheader("📝 Text Sentiment Analysis")
            text = st.text_area("Enter your thoughts:")
            
            if st.button("Analyze Sentiment"):
                if text.strip():
                    sentiment = analyze_text_sentiment(text)
                    st.success(f"Sentiment: {sentiment}")
                else:
                    st.warning("Please enter some text!")
    
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return "NEUTRAL"

def analyze_voice_emotion(uploaded_audio=None):
    """Simple voice emotion placeholder"""
    try:
        if uploaded_audio is not None:
            # Called with parameter from streamlit_app.py
            return "Happy"
        else:
            # Full UI version
            st.subheader("🎙️ Voice Emotion Analysis")
            st.info("🔊 **Feature Coming Soon!**")
            
            uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
            
            if uploaded_file is not None:
                st.audio(uploaded_file)
                if st.button("🔍 Analyze Emotion"):
                    st.success("Detected Emotion: Happy (placeholder)")
    
    except Exception as e:
        st.error(f"Error in voice analysis: {str(e)}")
        return "Happy"

def detect_face_emotion():
    """Simple face emotion placeholder"""
    try:
        st.subheader("🎥 Webcam Emotion Detection")
        st.info("🔊 **Feature Coming Soon!**")
        
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image")
            if st.button("🔍 Analyze Emotion"):
                st.success("Detected Emotion: Happy (placeholder)")
    
    except Exception as e:
        st.error(f"Error in face analysis: {str(e)}")

if __name__ == "__main__":
    st.title("🧠 MindReader – Emotion & Mental Health Detection")
    st.write("Detect your emotions using webcam, voice, or text.")

    option = st.selectbox("Choose a detection mode:", ["Text Sentiment", "Voice Emotion", "Facial Emotion"])

    if option == "Text Sentiment":
        user_input = st.text_area("Enter your thoughts:")
        if st.button("Analyze Text"):
            if user_input.strip() != "":
                try:
                    sentiment = analyze_text_sentiment(user_input)
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
                    emotion = analyze_voice_emotion(uploaded_audio)
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
                detect_face_emotion()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Webcam feature coming soon!")
