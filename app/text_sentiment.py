
import streamlit as st
import re

# Simple sentiment analysis using keyword-based approach
def simple_sentiment_analysis(text):
    # Convert to lowercase for analysis
    text = text.lower()
    
    # Positive words
    positive_words = ['happy', 'joy', 'love', 'excellent', 'good', 'great', 'wonderful', 
                     'amazing', 'fantastic', 'positive', 'cheerful', 'excited', 'glad',
                     'pleased', 'satisfied', 'delighted', 'thrilled', 'content', 'blessed']
    
    # Negative words  
    negative_words = ['sad', 'angry', 'hate', 'terrible', 'bad', 'awful', 'horrible',
                     'disgusting', 'negative', 'depressed', 'anxious', 'worried', 'upset',
                     'frustrated', 'disappointed', 'miserable', 'stressed', 'annoyed']
    
    # Count positive and negative words
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    # Determine sentiment
    if positive_count > negative_count:
        return "POSITIVE", (positive_count / (positive_count + negative_count + 1))
    elif negative_count > positive_count:
        return "NEGATIVE", (negative_count / (positive_count + negative_count + 1))
    else:
        return "NEUTRAL", 0.5

def analyze_text_sentiment():
    st.subheader("📝 Text Sentiment Analysis")
    text = st.text_area("Enter your thoughts or journal entry:", height=150)
    
    if st.button("Analyze Sentiment"):
        if text.strip():
            sentiment, confidence = simple_sentiment_analysis(text)
            
            # Display results
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
            
            # Show word count
            word_count = len(text.split())
            st.info(f"📊 Word count: {word_count}")
            
        else:
            st.warning("Please enter some text to analyze!")
