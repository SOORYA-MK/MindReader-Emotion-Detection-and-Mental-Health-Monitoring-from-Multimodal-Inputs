import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from deepface import DeepFace
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def detect_face_emotion():
    st.subheader("🎥 Webcam Emotion Detection")
    
    # Initialize session state for emotion history
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera Feed")
        
        # Camera controls
        camera_col1, camera_col2 = st.columns(2)
        with camera_col1:
            start_camera = st.button("🎥 Start Camera", key="start")
        with camera_col2:
            stop_camera = st.button("⏹️ Stop Camera", key="stop")
        
        # Placeholder for camera feed
        camera_placeholder = st.empty()
        
        # Image upload option as alternative
        st.markdown("### 📸 Or Upload an Image")
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Process uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Analyze Emotion in Image"):
                with st.spinner("Analyzing emotion..."):
                    try:
                        # Convert PIL image to numpy array
                        img_array = np.array(image)
                        
                        # Analyze emotion using DeepFace
                        result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
                        
                        if isinstance(result, list):
                            result = result[0]
                        
                        # Get dominant emotion
                        emotions = result['emotion']
                        dominant_emotion = result['dominant_emotion']
                        
                        # Store in history
                        st.session_state.emotion_history.append({
                            'timestamp': datetime.now(),
                            'emotion': dominant_emotion,
                            'confidence': emotions[dominant_emotion],
                            'all_emotions': emotions
                        })
                        
                        # Display results immediately after col1
                        st.success("✅ Analysis Complete!")
                        
                    except Exception as e:
                        st.error(f"Error analyzing emotion: {str(e)}")
                        st.info("💡 Try uploading a clear image with a visible face")
    
    with col2:
        st.markdown("### 📊 Emotion Analysis")
        
        if st.session_state.emotion_history:
            # Get the latest analysis
            latest = st.session_state.emotion_history[-1]
            emotions = latest['all_emotions']
            dominant_emotion = latest['emotion']
            
            # Show dominant emotion with emoji
            emotion_emoji = {
                'happy': '😊', 'sad': '😢', 'angry': '😠', 
                'fear': '😨', 'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
            }
            
            emoji = emotion_emoji.get(dominant_emotion.lower(), '😐')
            st.success(f"{emoji} **{dominant_emotion.title()}**")
            st.metric("Confidence", f"{emotions[dominant_emotion]:.1f}%")
            
            # Show all emotions as a bar chart
            st.markdown("### 📈 All Emotions")
            emotion_df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Score'])
            emotion_df['Score'] = emotion_df['Score'].round(1)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(emotion_df['Emotion'], emotion_df['Score'])
            ax.set_xlabel('Confidence (%)')
            ax.set_title('Emotion Analysis Results')
            
            # Color bars based on emotion
            colors = ['#ff6b6b' if e == dominant_emotion else '#95a5a6' for e in emotion_df['Emotion']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("### 📈 Emotion History")
        
        if st.session_state.emotion_history:
            # Show recent emotions
            recent_emotions = st.session_state.emotion_history[-5:]  # Last 5 entries
            
            # Create a simple timeline
            for i, entry in enumerate(reversed(recent_emotions)):
                emotion = entry['emotion']
                confidence = entry['confidence']
                timestamp = entry['timestamp'].strftime("%H:%M:%S")
                
                emoji = {
                    'happy': '😊', 'sad': '😢', 'angry': '😠', 
                    'fear': '😨', 'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
                }.get(emotion.lower(), '😐')
                
                st.write(f"{emoji} **{emotion.title()}** ({confidence:.1f}%) - {timestamp}")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.emotion_history = []
                st.rerun()
                
        else:
            st.info("No emotion data yet. Upload an image to start!")
    
    # Instructions
    with st.expander("📋 Instructions", expanded=False):
        st.write("""
        **How to use:**
        1. **Upload Image**: Click "Choose an image file" and select a photo with a clear face
        2. **Analyze**: Click "Analyze Emotion in Image" to detect emotions
        3. **View Results**: See the dominant emotion and confidence scores
        4. **History**: Track your emotion analysis over time
        
        **Tips for better results:**
        - Use clear, well-lit photos
        - Ensure the face is clearly visible
        - Avoid sunglasses or face coverings
        - Front-facing photos work best
        
        **Note**: Live camera functionality requires additional setup for security permissions.
        """)
    
    # Technical info
    with st.expander("🔧 Technical Details", expanded=False):
        st.write("""
        **Technology Stack:**
        - **DeepFace**: Advanced facial emotion recognition
        - **OpenCV**: Image processing and computer vision
        - **Deep Learning**: CNN-based emotion classification
        
        **Supported Emotions:**
        - 😊 Happy - Joy, contentment, satisfaction
        - 😢 Sad - Sorrow, melancholy, disappointment  
        - 😠 Angry - Frustration, irritation, rage
        - 😨 Fear - Anxiety, worry, apprehension
        - 😲 Surprise - Amazement, shock, wonder
        - 🤢 Disgust - Revulsion, dislike, aversion
        - 😐 Neutral - Calm, composed, indifferent
        """)
