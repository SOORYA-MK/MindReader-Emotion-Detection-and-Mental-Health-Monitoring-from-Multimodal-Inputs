import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import hashlib

# Try to import DeepFace for real emotion detection
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

def real_emotion_detection(image_array):
    """
    Real emotion detection using DeepFace (trained on FER2013 dataset)
    Falls back to improved rule-based detection if DeepFace is not available
    """
    if DEEPFACE_AVAILABLE:
        try:
            # Convert numpy array to PIL Image and save temporarily
            img = Image.fromarray(image_array)
            temp_path = "temp_image.jpg"
            img.save(temp_path)
            
            # Analyze emotion using DeepFace (trained on real datasets)
            result = DeepFace.analyze(temp_path, actions=['emotion'], enforce_detection=False)
            
            # Extract emotions from result
            if isinstance(result, list):
                emotions_dict = result[0]['emotion']
            else:
                emotions_dict = result['emotion']
            
            # Get dominant emotion
            dominant_emotion = max(emotions_dict, key=emotions_dict.get)
            
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return emotions_dict, dominant_emotion
            
        except Exception as e:
            st.warning(f"DeepFace analysis failed: {str(e)}. Using fallback method.")
            return improved_rule_based_detection(image_array)
    else:
        return improved_rule_based_detection(image_array)

def improved_rule_based_detection(image_array):
    """
    Improved rule-based emotion detection as fallback
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Load OpenCV's pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    
    if len(faces) == 0:
        return None, "No face detected"
    
    # Get the largest face (main subject)
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    
    # Extract face region
    face_roi = gray[y:y+h, x:x+w]
    
    if face_roi.size == 0:
        return None, "Face region too small"
    
    # Create hash for consistency
    face_bytes = face_roi.tobytes()
    image_hash = hashlib.md5(face_bytes).hexdigest()
    hash_int = int(image_hash[:8], 16)
    
    # More sophisticated facial analysis
    face_height, face_width = face_roi.shape
    
    # Define regions more precisely
    eye_upper = face_roi[int(face_height * 0.2):int(face_height * 0.4), :]
    eye_lower = face_roi[int(face_height * 0.35):int(face_height * 0.5), :]
    nose_region = face_roi[int(face_height * 0.35):int(face_height * 0.65), 
                         int(face_width * 0.3):int(face_width * 0.7)]
    mouth_region = face_roi[int(face_height * 0.6):int(face_height * 0.9), 
                          int(face_width * 0.2):int(face_width * 0.8)]
    
    # Calculate advanced metrics
    overall_brightness = np.mean(face_roi)
    overall_contrast = np.std(face_roi)
    
    # Eye metrics
    eye_brightness = np.mean(eye_upper) if eye_upper.size > 0 else overall_brightness
    eye_openness = np.std(eye_upper) if eye_upper.size > 0 else 0
    
    # Mouth metrics  
    mouth_brightness = np.mean(mouth_region) if mouth_region.size > 0 else overall_brightness
    mouth_curve = 0
    if mouth_region.size > 0:
        mouth_top = np.mean(mouth_region[:mouth_region.shape[0]//3, :])
        mouth_bottom = np.mean(mouth_region[mouth_region.shape[0]*2//3:, :])
        mouth_curve = mouth_bottom - mouth_top  # Positive for smile, negative for frown
    
    # Edge detection for expression lines
    edges = cv2.Canny(face_roi, 50, 150)
    edge_density = np.sum(edges > 0) / (face_height * face_width)
    
    # Forehead tension (wrinkles)
    forehead = face_roi[:int(face_height * 0.3), :]
    forehead_roughness = np.std(forehead) if forehead.size > 0 else 0
    
    # Emotion scoring with better logic
    np.random.seed(hash_int % 1000)  # For slight variations
    
    # Base scores
    emotions = {}
    
    # ANGRY: Dark eyes, frown, forehead tension, high contrast
    angry_indicators = [
        max(0, overall_brightness - eye_brightness) / 50,  # Dark eyes
        max(0, -mouth_curve) / 20,  # Frown
        forehead_roughness / 100,  # Forehead tension
        overall_contrast / 100,  # High contrast
        edge_density * 200  # Sharp edges
    ]
    emotions['angry'] = sum(angry_indicators) * 20 + np.random.uniform(-5, 5)
    
    # HAPPY: Bright mouth, smile curve, moderate eyes
    happy_indicators = [
        max(0, mouth_brightness - overall_brightness) / 30,  # Bright mouth
        max(0, mouth_curve) / 15,  # Smile curve
        (eye_brightness / 255) * 0.3,  # Not too dark eyes
        max(0, 1 - forehead_roughness / 50)  # Relaxed forehead
    ]
    emotions['happy'] = sum(happy_indicators) * 25 + np.random.uniform(-3, 3)
    
    # SAD: Dark overall, downturned mouth, low contrast
    sad_indicators = [
        max(0, 150 - overall_brightness) / 100,  # Dark overall
        max(0, -mouth_curve) / 25,  # Downturned mouth
        max(0, 1 - overall_contrast / 100),  # Low contrast
        max(0, overall_brightness - mouth_brightness) / 40  # Dark mouth
    ]
    emotions['sad'] = sum(sad_indicators) * 30 + np.random.uniform(-4, 4)
    
    # SURPRISE: Wide eyes, open mouth, high contrast
    surprise_indicators = [
        eye_openness / 50,  # Wide eyes
        mouth_brightness / 255 * 0.5,  # Bright/open mouth
        overall_contrast / 100  # High contrast
    ]
    emotions['surprise'] = sum(surprise_indicators) * 20 + np.random.uniform(-3, 3)
    
    # FEAR: Dark, tense, high contrast
    fear_indicators = [
        max(0, 120 - overall_brightness) / 80,  # Dark
        forehead_roughness / 80,  # Tension
        overall_contrast / 120  # High contrast
    ]
    emotions['fear'] = sum(fear_indicators) * 15 + np.random.uniform(-2, 2)
    
    # DISGUST: Nose wrinkles, mouth tension
    disgust_indicators = [
        forehead_roughness / 100,  # Wrinkles
        abs(mouth_curve) / 30,  # Mouth tension
        edge_density * 100  # Sharp features
    ]
    emotions['disgust'] = sum(disgust_indicators) * 12 + np.random.uniform(-2, 2)
    
    # NEUTRAL: Balanced features
    neutral_score = 20 - sum([
        abs(mouth_curve) / 20,  # Not too much curve
        abs(overall_brightness - 128) / 100,  # Moderate brightness  
        overall_contrast / 200,  # Not too much contrast
        forehead_roughness / 100  # Not too much tension
    ]) + np.random.uniform(-3, 3)
    emotions['neutral'] = max(5, neutral_score)
    
    # Ensure all scores are positive
    for emotion in emotions:
        emotions[emotion] = max(0.1, emotions[emotion])
    
    # Normalize to 100%
    total = sum(emotions.values())
    emotions = {k: (v/total)*100 for k, v in emotions.items()}
    
    # Get dominant emotion
    dominant_emotion = max(emotions, key=emotions.get)
    
    return emotions, dominant_emotion

def detect_face_emotion():
    st.subheader("🎥 Webcam Emotion Detection")
    
    # Show which method is being used
    if DEEPFACE_AVAILABLE:
        st.success("✅ Using DeepFace AI model (trained on FER2013 dataset)")
    else:
        st.warning("⚠️ Using rule-based detection. Install DeepFace for better accuracy: `pip install deepface`")
    
    # Initialize session state
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    
    if 'analyzed_images' not in st.session_state:
        st.session_state.analyzed_images = {}
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Upload an Image for Emotion Analysis")
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Process uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Create hash for consistency
            image_bytes = uploaded_file.getvalue()
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            # Analysis options
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                analyze_btn = st.button("🔍 Analyze Emotion", use_container_width=True)
            
            with col_btn2:
                force_analyze_btn = st.button("🔄 Force Re-analyze", use_container_width=True, 
                                            help="Re-analyze even if image was previously analyzed")
            
            if analyze_btn or force_analyze_btn:
                # Check if we should use cache or force new analysis
                use_cache = not force_analyze_btn and image_hash in st.session_state.analyzed_images
                
                if use_cache:
                    cached_result = st.session_state.analyzed_images[image_hash]
                    emotions = cached_result['emotions']
                    dominant_emotion = cached_result['dominant_emotion']
                    st.info("🔄 Using cached analysis (same image detected)")
                else:
                    # Perform new analysis
                    with st.spinner("🧠 Analyzing facial emotions with AI..."):
                        try:
                            # Convert to numpy array
                            img_array = np.array(image)
                            
                            # Analyze emotion with real AI model
                            emotions, dominant_emotion = real_emotion_detection(img_array)
                            
                            if emotions is None:
                                st.error("❌ " + dominant_emotion)
                                st.info("💡 Try uploading a clearer image with a visible face")
                                return
                            
                            # Cache the result (or update if forced)
                            st.session_state.analyzed_images[image_hash] = {
                                'emotions': emotions,
                                'dominant_emotion': dominant_emotion,
                                'timestamp': datetime.now(),
                                'method': 'DeepFace' if DEEPFACE_AVAILABLE else 'Rule-based'
                            }
                            
                            if force_analyze_btn:
                                st.success("✅ Fresh AI analysis completed!")
                            else:
                                st.success("✅ AI analysis completed!")
                            
                        except Exception as e:
                            st.error(f"❌ Error analyzing emotion: {str(e)}")
                            st.info("💡 Please try a different image or check image format")
                            return
                
                # Store in history
                st.session_state.emotion_history.append({
                    'timestamp': datetime.now(),
                    'emotion': dominant_emotion,
                    'confidence': emotions[dominant_emotion],
                    'all_emotions': emotions,
                    'image_hash': image_hash,
                    'method': 'DeepFace' if DEEPFACE_AVAILABLE else 'Rule-based'
                })
        
        # Live camera section
        st.markdown("### 🎥 Live Camera (Coming Soon)")
        st.info("🚧 Real-time webcam functionality will be available in a future update.")
    
    with col2:
        st.markdown("### 📊 Current Analysis")
        
        if st.session_state.emotion_history:
            # Get latest analysis
            latest = st.session_state.emotion_history[-1]
            emotions = latest['all_emotions']
            dominant_emotion = latest['emotion']
            method = latest.get('method', 'Unknown')
            
            # Show dominant emotion
            emotion_emoji = {
                'happy': '😊', 'sad': '😢', 'angry': '😠', 
                'fear': '😨', 'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
            }
            
            emoji = emotion_emoji.get(dominant_emotion.lower(), '😐')
            st.success(f"{emoji} **{dominant_emotion.title()}**")
            st.metric("Confidence", f"{emotions[dominant_emotion]:.1f}%")
            st.caption(f"Method: {method}")
            
            # Emotion breakdown chart
            st.markdown("### 📈 Emotion Breakdown")
            emotion_df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Score'])
            emotion_df = emotion_df.sort_values('Score', ascending=True)
            emotion_df['Score'] = emotion_df['Score'].round(1)
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(emotion_df['Emotion'], emotion_df['Score'])
            ax.set_xlabel('Confidence (%)')
            ax.set_title('Emotion Analysis Results')
            ax.grid(True, alpha=0.3)
            
            # Color the dominant emotion differently
            colors = ['#ff6b6b' if e == dominant_emotion else '#95a5a6' for e in emotion_df['Emotion']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Upload an image to see emotion analysis results")
        
        # Emotion history
        st.markdown("### 📈 Recent History")
        
        if st.session_state.emotion_history:
            recent_emotions = st.session_state.emotion_history[-5:]
            
            for entry in reversed(recent_emotions):
                emotion = entry['emotion']
                confidence = entry['confidence']
                timestamp = entry['timestamp'].strftime("%H:%M:%S")
                method = entry.get('method', '')
                
                emoji = {
                    'happy': '😊', 'sad': '😢', 'angry': '😠', 
                    'fear': '😨', 'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
                }.get(emotion.lower(), '😐')
                
                st.write(f"{emoji} **{emotion.title()}** ({confidence:.1f}%) - {timestamp}")
                if method:
                    st.caption(f"   └─ {method}")
            
            if st.button("🗑️ Clear History"):
                st.session_state.emotion_history = []
                st.session_state.analyzed_images = {}
                st.rerun()
        else:
            st.info("No analysis history yet")
    
    # Debug information
    if st.session_state.emotion_history and st.checkbox("🔍 Show Analysis Details"):
        st.markdown("### 🛠️ Analysis Details")
        latest = st.session_state.emotion_history[-1]
        st.write("**Latest analysis breakdown:**")
        
        emotions = latest['all_emotions']
        method = latest.get('method', 'Unknown')
        
        st.info(f"**Analysis Method**: {method}")
        
        if method == 'DeepFace':
            st.write("🤖 **DeepFace AI Model**: Trained on FER2013 dataset with 35,887 facial images")
        else:
            st.write("📊 **Rule-based Algorithm**: Uses facial feature analysis")
        
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, score in sorted_emotions:
            if score > 1:  # Show all significant scores
                emoji = {
                    'happy': '😊', 'sad': '😢', 'angry': '😠', 
                    'fear': '😨', 'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
                }.get(emotion.lower(), '😐')
                
                color = "🔴" if emotion == latest['emotion'] else "⚪"
                st.write(f"{color} {emoji} **{emotion.title()}**: {score:.1f}%")
    
    # Instructions
    with st.expander("📋 How to Use", expanded=False):
        st.write("""
        **Steps:**
        1. **Upload Image**: Click "Choose an image file" and select a photo
        2. **Analyze**: Click "🔍 Analyze Emotion" for AI analysis
        3. **Re-analyze**: Click "🔄 Force Re-analyze" to get fresh results
        4. **View Results**: See emotion analysis in the right panel
        5. **Debug**: Check "🔍 Show Analysis Details" to see method used
        
        **Dataset Information:**
        - **DeepFace**: Uses pre-trained models from FER2013 dataset (35,887 facial images)
        - **FER2013**: Contains 7 emotion categories trained on thousands of real faces
        - **Real AI**: Much more accurate than rule-based detection
        
        **Tips for Better Results:**
        - 📸 Use clear, well-lit photos with visible faces
        - 👤 Ensure face is front-facing and unobstructed
        - 😠 The AI model is trained on real angry expressions
        - 🚫 Avoid sunglasses, masks, or heavy shadows
        
        **If results seem wrong:**
        - Try the "🔄 Force Re-analyze" button
        - Check if DeepFace AI model is being used (green message at top)
        - Ensure good lighting and face visibility
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details", expanded=False):
        st.write(f"""
        **Current Method**: {'DeepFace AI Model' if DEEPFACE_AVAILABLE else 'Rule-based Fallback'}
        
        **DeepFace AI Features:**
        - 🤖 **Pre-trained Models**: Uses state-of-the-art deep learning models
        - 📊 **FER2013 Dataset**: Trained on 35,887 real facial emotion images
        - 🎯 **7 Emotions**: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
        - 🔬 **Research-based**: Uses proven computer vision techniques
        - 📈 **High Accuracy**: Much better than rule-based approaches
        
        **Why DeepFace is Better:**
        - ✅ Trained on real human facial expressions
        - ✅ Uses deep learning neural networks
        - ✅ Recognizes subtle facial features
        - ✅ Handles various lighting conditions
        - ✅ More accurate emotion detection
        
        **Dataset Source:**
        - FER2013: Facial Expression Recognition Challenge dataset
        - Contains thousands of labeled facial expressions
        - Used to train state-of-the-art emotion recognition models
        """)
