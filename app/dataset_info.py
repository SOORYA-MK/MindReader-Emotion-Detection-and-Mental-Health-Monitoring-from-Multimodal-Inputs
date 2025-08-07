import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

def show_dataset_info():
    """
    Show detailed information about emotion detection datasets and how they work
    """
    st.header("📊 Emotion Detection Dataset Information")
    
    # Dataset overview
    st.markdown("## 🗃️ FER2013 Dataset Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 Dataset Statistics
        - **Total Images**: 35,887 grayscale images
        - **Image Size**: 48x48 pixels
        - **Classes**: 7 emotions
        - **Split**: Train/Validation/Test
        - **Source**: Faces collected from Google Image Search
        - **Labels**: Human-annotated emotions
        """)
        
        # Create dataset statistics
        fer_stats = {
            'Emotion': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
            'Training Samples': [4953, 547, 5121, 8989, 6077, 4002, 6198],
            'Percentage': [13.8, 1.5, 14.3, 25.1, 16.9, 11.2, 17.3]
        }
        
        df_stats = pd.DataFrame(fer_stats)
        st.dataframe(df_stats, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### 🎯 Key Insights About The Dataset
        - **Happy** has the most samples (25.1%)
        - **Disgust** has the least samples (1.5%)
        - **Imbalanced dataset** - some emotions better trained
        - **Low resolution** (48x48) affects accuracy
        - **Gray-scale only** - no color information
        - **Crowd-sourced** labels can be inconsistent
        """)
        
        # Create pie chart of emotion distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#a0a0a0']
        ax.pie(df_stats['Training Samples'], labels=df_stats['Emotion'], colors=colors, autopct='%1.1f%%')
        ax.set_title('FER2013 Dataset Distribution')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Dataset problems
    st.markdown("## ⚠️ Why Emotion Detection Can Be Inaccurate")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Issues", "Model Limitations", "Real Examples", "Better Alternatives"])
    
    with tab1:
        st.markdown("""
        ### 🚨 FER2013 Dataset Problems
        
        **1. Data Quality Issues:**
        - Many images are **mislabeled** by human annotators
        - **Subjective emotions** - same face can be multiple emotions
        - **Context missing** - no background or situation context
        - **Cultural bias** - mostly Western faces
        
        **2. Technical Limitations:**
        - **Low resolution** (48x48) loses facial details
        - **Grayscale only** - no skin color/lip color information
        - **Single frame** - no temporal information
        - **Posed vs Natural** - many are posed/acted emotions
        
        **3. Class Imbalance:**
        - **Happy**: 8,989 samples (well-trained)
        - **Disgust**: 547 samples (poorly-trained)
        - **Sad/Crying**: Often confused with each other
        - **Angry**: Sometimes labeled as disgusted or sad
        """)
        
        # Show the actual distribution problem
        emotion_accuracy = {
            'Emotion': ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Neutral', 'Disgust'],
            'Expected Accuracy': [85, 65, 70, 75, 60, 80, 45],
            'Reason': [
                'Most samples, clear expression',
                'Often confused with neutral/angry',
                'Confused with disgust/sad',
                'Usually distinct features',
                'Similar to sad/surprise',
                'Well-balanced, common state',
                'Least samples, subtle expression'
            ]
        }
        
        acc_df = pd.DataFrame(emotion_accuracy)
        st.dataframe(acc_df, use_container_width=True)
    
    with tab2:
        st.markdown("""
        ### 🤖 Model Architecture Limitations
        
        **DeepFace Uses Multiple Models:**
        1. **VGG-Face**: Good for face recognition, basic emotions
        2. **OpenFace**: Lightweight but less accurate
        3. **FaceNet**: Better for face verification than emotion
        4. **DeepID**: Older architecture
        
        **Why Crying → Sad Detection:**
        - **Facial Action Units**: Crying and sadness share similar muscle movements
        - **Tear detection**: Models don't specifically look for tears
        - **Context missing**: Can't tell if tears are from joy, sadness, or pain
        - **Training bias**: More "sad" labeled faces than "crying" faces
        
        **Technical Reasons:**
        ```
        Crying face features detected:
        - Drooped eyelids → Classified as "Sad"
        - Downturned mouth → Classified as "Sad"  
        - Tensed facial muscles → Could be "Angry" or "Sad"
        - Tears (if visible) → Not specifically trained to detect
        ```
        """)
        
    with tab3:
        st.markdown("""
        ### 🔍 Real Examples of Dataset Issues
        
        **Common Misclassifications:**
        
        | Your Input | Model Says | Why Wrong | Actual Issue |
        |------------|------------|-----------|--------------|
        | 😭 Crying face | 😢 Sad | Tears not detected | Dataset has few crying examples |
        | 😠 Angry face | 😊 Happy | Mouth shape confusion | Lighting/angle issues |
        | 😐 Neutral thinking | 😢 Sad | Slight frown detected | Over-sensitive to mouth position |
        | 🤔 Contemplative | 😠 Angry | Brow furrow detected | Thinking confused with anger |
        
        **Why This Happens:**
        - **Limited training data** for subtle expressions
        - **Human labeling errors** in original dataset
        - **Cultural differences** in expression interpretation
        - **Individual variation** in how people express emotions
        """)
        
        # Show confusion matrix simulation
        st.markdown("#### Typical Confusion Matrix")
        confusion_data = np.array([
            [70, 5, 10, 2, 8, 3, 2],    # Angry
            [8, 45, 5, 3, 15, 4, 20],   # Disgust  
            [12, 3, 60, 5, 15, 3, 2],   # Fear
            [3, 1, 2, 85, 4, 3, 2],     # Happy
            [15, 8, 8, 5, 65, 2, 7],    # Sad
            [5, 2, 8, 12, 3, 75, 5],    # Surprise
            [10, 5, 2, 8, 12, 3, 80]    # Neutral
        ])
        
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotions, yticklabels=emotions, ax=ax)
        ax.set_title('Typical Emotion Detection Confusion Matrix')
        ax.set_xlabel('Predicted Emotion')
        ax.set_ylabel('True Emotion')
        st.pyplot(fig)
    
    with tab4:
        st.markdown("""
        ### 🚀 Better Alternatives & Solutions
        
        **1. More Advanced Datasets:**
        - **AffectNet**: 1M+ images, better quality
        - **RAF-DB**: Real-world faces, better labeling
        - **EmotiW**: Video-based with context
        - **CK+**: Laboratory controlled expressions
        
        **2. Multimodal Approaches:**
        - **Text + Image**: Analyze context from captions/chat
        - **Video**: Temporal information from multiple frames
        - **Audio + Visual**: Voice tone + facial expression
        - **Physiological**: Heart rate, skin conductance
        
        **3. Custom Training:**
        - **Domain-specific**: Train on your specific use case
        - **Cultural adaptation**: Train on diverse populations
        - **Context-aware**: Include background/situation
        - **Personal calibration**: Learn individual expression patterns
        
        **4. Ensemble Methods:**
        - **Multiple models**: Combine different architectures
        - **Confidence scoring**: Only trust high-confidence predictions
        - **Human-in-the-loop**: Ask for verification on uncertain cases
        """)
        
        # Show better model comparison
        model_comparison = {
            'Model/Dataset': ['FER2013 (Current)', 'AffectNet', 'RAF-DB', 'Custom Ensemble'],
            'Accuracy': ['65%', '78%', '82%', '85%+'],
            'Dataset Size': ['36K', '1M+', '30K', 'Variable'],
            'Quality': ['Poor', 'Good', 'Excellent', 'Custom'],
            'Cost': ['Free', 'Academic', 'Research', 'High']
        }
        
        comp_df = pd.DataFrame(model_comparison)
        st.dataframe(comp_df, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("## 💡 Recommendations for Your Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        ### ✅ Short-term Improvements
        1. **Add confidence thresholds** - only show results >70% confidence
        2. **Multiple model ensemble** - use 2-3 different models
        3. **User feedback** - let users correct wrong predictions
        4. **Context questions** - ask "Are you crying from sadness or joy?"
        """)
    
    with col2:
        st.info("""
        ### 🔮 Long-term Solutions
        1. **Better dataset** - use AffectNet or RAF-DB
        2. **Video analysis** - analyze multiple frames
        3. **Multimodal** - combine text, voice, and face
        4. **Custom training** - train on your specific domain
        """)
    
    # Code example for improvement
    with st.expander("🔧 Code Example: Confidence-based Filtering"):
        st.code("""
def improved_emotion_detection(image):
    # Get predictions from multiple models
    results = []
    
    # Model 1: DeepFace
    result1 = DeepFace.analyze(image, actions=['emotion'])
    results.append(result1['emotion'])
    
    # Model 2: FER (different architecture)
    result2 = fer_model.predict(image)
    results.append(result2)
    
    # Ensemble: Average predictions
    ensemble_result = average_predictions(results)
    
    # Apply confidence threshold
    max_confidence = max(ensemble_result.values())
    
    if max_confidence < 0.7:  # 70% threshold
        return "Uncertain - please verify", ensemble_result
    else:
        return max(ensemble_result, key=ensemble_result.get), ensemble_result
        
def handle_crying_detection(emotions):
    # Special case: if sad + high eye region activity
    if emotions['sad'] > 0.6 and eye_activity_high():
        return "crying", emotions
    return emotions
        """, language='python')

if __name__ == "__main__":
    show_dataset_info()
