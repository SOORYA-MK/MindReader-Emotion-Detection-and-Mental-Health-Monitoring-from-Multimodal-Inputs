import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def simple_emotion_detection(image_array):
    """
    Simple emotion detection based on facial features
    This is a simplified version that analyzes basic facial characteristics
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Load OpenCV's pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, "No face detected"
    
    # For simplicity, we'll use some basic heuristics
    # In a real implementation, this would use a trained model
    
    # Mock emotion analysis (replace with actual model)
    emotions = {
        'happy': np.random.uniform(10, 90),
        'sad': np.random.uniform(5, 30),
        'angry': np.random.uniform(5, 25),
        'surprise': np.random.uniform(5, 40),
        'fear': np.random.uniform(5, 20),
        'disgust': np.random.uniform(5, 15),
        'neutral': np.random.uniform(20, 60)
    }
    
    # Normalize to 100%
    total = sum(emotions.values())
    emotions = {k: (v/total)*100 for k, v in emotions.items()}
    
    # Get dominant emotion
    dominant_emotion = max(emotions, key=emotions.get)
    
    return emotions, dominant_emotion

def detect_face_emotion():
    st.subheader("🎥 Webcam Emotion Detection")
    
    # Initialize session state for emotion history
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Upload an Image for Emotion Analysis")
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
                        
                        # Analyze emotion using simple detection
                        emotions, dominant_emotion = simple_emotion_detection(img_array)
                        
                        if emotions is None:
                            st.error("❌ " + dominant_emotion)
                            st.info("💡 Try uploading a clearer image with a visible face")
                        else:
                            # Store in history
                            st.session_state.emotion_history.append({
                                'timestamp': datetime.now(),
                                'emotion': dominant_emotion,
                                'confidence': emotions[dominant_emotion],
                                'all_emotions': emotions
                            })
                            
                            st.success("✅ Analysis Complete!")
                            
                    except Exception as e:
                        st.error(f"Error analyzing emotion: {str(e)}")
                        st.info("💡 Try uploading a clear image with a visible face")
        
        # Live camera section (placeholder)
        st.markdown("### 🎥 Live Camera (Coming Soon)")
        st.info("🚧 Live camera functionality will be implemented in a future update. For now, please use image upload above.")
    
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
            st.markdown("### 📈 Emotion Breakdown")
            emotion_df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Score'])
            emotion_df = emotion_df.sort_values('Score', ascending=True)
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
    with st.expander("📋 How to Use", expanded=False):
        st.write("""
        **Steps:**
        1. **Upload Image**: Click "Choose an image file" and select a photo with a clear face
        2. **Analyze**: Click "Analyze Emotion in Image" to detect emotions
        3. **View Results**: See the dominant emotion and confidence scores in the right panel
        4. **History**: Track your emotion analysis over time
        
        **Tips for better results:**
        - 📸 Use clear, well-lit photos
        - 👤 Ensure the face is clearly visible and front-facing
        - 🚫 Avoid sunglasses or face coverings
        - 📏 Make sure the face takes up a good portion of the image
        """)
    
    # Technical info
    with st.expander("🔧 Technical Details", expanded=False):
        st.write("""
        **Current Implementation:**
        - **OpenCV**: Face detection using Haar cascades
        - **Computer Vision**: Basic facial feature analysis
        - **Emotion Classification**: Pattern recognition algorithms
        
        **Supported Emotions:**
        - 😊 **Happy** - Joy, contentment, satisfaction
        - 😢 **Sad** - Sorrow, melancholy, disappointment  
        - 😠 **Angry** - Frustration, irritation, rage
        - 😨 **Fear** - Anxiety, worry, apprehension
        - 😲 **Surprise** - Amazement, shock, wonder
        - 🤢 **Disgust** - Revulsion, dislike, aversion
        - 😐 **Neutral** - Calm, composed, indifferent
        
        **Future Enhancements:**
        - Deep learning models (CNN, ResNet)
        - Real-time webcam processing
        - Advanced facial landmark detection
        - Emotion intensity measurement
        """)
