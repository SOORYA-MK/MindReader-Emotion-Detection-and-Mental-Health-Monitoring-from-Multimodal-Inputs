# Simple text sentiment analysis for MindReader app

def simple_sentiment_analysis(text):
    """Basic keyword-based sentiment analysis"""
    if not text or not isinstance(text, str):
        return "NEUTRAL", 0.5
    
    text = text.lower()
    
    positive_words = ['happy', 'joy', 'love', 'excellent', 'good', 'great', 'wonderful', 
                     'amazing', 'fantastic', 'positive', 'cheerful', 'excited', 'glad']
    
    negative_words = ['sad', 'angry', 'hate', 'terrible', 'bad', 'awful', 'horrible',
                     'disgusting', 'negative', 'depressed', 'anxious', 'worried', 'upset']
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        return "POSITIVE", positive_count / (positive_count + negative_count + 1)
    elif negative_count > positive_count:
        return "NEGATIVE", negative_count / (positive_count + negative_count + 1)
    else:
        return "NEUTRAL", 0.5

def analyze_text_sentiment(user_input=None):
    """
    Analyze text sentiment - compatible with both parameter and UI calls
    """
    import streamlit as st
    
    if user_input is not None:
        # Called with parameter from streamlit_app.py
        try:
            sentiment, confidence = simple_sentiment_analysis(user_input)
            return sentiment
        except Exception:
            return "NEUTRAL"
    else:
        # Called from main app with full UI
        st.subheader("📝 Text Sentiment Analysis")
        text = st.text_area("Enter your thoughts or journal entry:", height=150)
        
        if st.button("Analyze Sentiment"):
            if text.strip():
                try:
                    sentiment, confidence = simple_sentiment_analysis(text)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if sentiment == "POSITIVE":
                            st.success(f"😊 **Sentiment**: {sentiment}")
                        elif sentiment == "NEGATIVE":
                            st.error(f"😞 **Sentiment**: {sentiment}")
                        else:
                            st.info(f"😐 **Sentiment**: {sentiment}")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}")
                    
                    word_count = len(text.split())
                    st.info(f"📊 Word count: {word_count}")
                    
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {str(e)}")
            else:
                st.warning("Please enter some text to analyze!")
