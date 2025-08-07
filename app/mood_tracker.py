
import streamlit as st
import matplotlib.pyplot as plt
import datetime
import json
import os

def load_mood_data():
    """Load mood data from a JSON file"""
    mood_file = "mood_data.json"
    if os.path.exists(mood_file):
        try:
            with open(mood_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_mood_data(data):
    """Save mood data to a JSON file"""
    mood_file = "mood_data.json"
    with open(mood_file, 'w') as f:
        json.dump(data, f)

def track_mood():
    st.subheader("📊 Mood Tracker")
    
    # Load existing mood data
    mood_data = load_mood_data()
    
    # Input section for new mood entry
    with st.expander("📝 Add New Mood Entry", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            mood_date = st.date_input("Date", datetime.date.today())
            
        with col2:
            mood_score = st.slider("Mood Score (1-10)", 1, 10, 5, 
                                 help="1 = Very Sad, 5 = Neutral, 10 = Very Happy")
        
        mood_note = st.text_area("Notes (optional)", height=100, 
                                placeholder="What influenced your mood today?")
        
        if st.button("Save Mood Entry"):
            date_str = mood_date.strftime("%Y-%m-%d")
            mood_data[date_str] = {
                "score": mood_score,
                "note": mood_note
            }
            save_mood_data(mood_data)
            st.success("Mood entry saved!")
            st.rerun()
    
    # Display mood history
    if mood_data:
        st.subheader("📈 Your Mood Timeline")
        
        # Prepare data for plotting
        dates = []
        scores = []
        
        # Sort dates and get recent entries
        sorted_dates = sorted(mood_data.keys(), reverse=True)[:14]  # Last 14 days
        sorted_dates.reverse()  # Oldest to newest for chart
        
        for date_str in sorted_dates:
            dates.append(datetime.datetime.strptime(date_str, "%Y-%m-%d").date())
            scores.append(mood_data[date_str]["score"])
        
        # Create chart
        if len(dates) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(dates, scores, marker='o', linewidth=2, markersize=6)
            ax.set_ylabel('Mood Score')
            ax.set_title('Mood Over Time')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 11)
            
            # Color code the line based on mood
            for i, score in enumerate(scores):
                color = 'red' if score <= 3 else 'orange' if score <= 6 else 'green'
                ax.plot(dates[i], score, 'o', color=color, markersize=8)
            
            st.pyplot(fig)
            
            # Show statistics
            avg_mood = sum(scores) / len(scores)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Mood", f"{avg_mood:.1f}")
            with col2:
                st.metric("Best Day", f"{max(scores)}")
            with col3:
                st.metric("Lowest Day", f"{min(scores)}")
        
        # Show recent entries
        st.subheader("📝 Recent Entries")
        for date_str in sorted(mood_data.keys(), reverse=True)[:5]:
            entry = mood_data[date_str]
            with st.expander(f"{date_str} - Mood: {entry['score']}/10"):
                if entry.get('note'):
                    st.write(f"**Note:** {entry['note']}")
                else:
                    st.write("*No notes for this day*")
    else:
        st.info("🌟 Start tracking your mood by adding your first entry above!")
