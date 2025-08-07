![GitHub Repo stars](https://img.shields.io/github/stars/SOORYA-MK/MindReader-Emotion-Detection-and-Mental-Health-Monitoring-from-Multimodal-Inputs)
![GitHub forks](https://img.shields.io/github/forks/SOORYA-MK/MindReader-Emotion-Detection-and-Mental-Health-Monitoring-from-Multimodal-Inputs)
![GitHub license](https://img.shields.io/github/license/SOORYA-MK/MindReader-Emotion-Detection-and-Mental-Health-Monitoring-from-Multimodal-Inputs)

# MindReader – Emotion & Mental Health Detection from Multimodal Inputs

**MindReader** is an AI-powered mental health and emotion detection system that uses **facial expressions**, **voice tone**, and **text sentiment** to identify the emotional state of a user in real-time.

## 🔍 Features

- 🎥 **Facial Emotion Detection** (via webcam)
- 🎙️ **Voice Emotion Analysis** (from microphone or audio file)
- ✍️ **Text Sentiment Analysis** (using BERT)
- 📊 **Mood Tracking Dashboard** (coming soon)
- 🧠 Multi-modal integration for more accurate mental state analysis

## 🚀 Tech Stack

- Python 3.10+
- OpenCV, TensorFlow, DeepFace, FER
- BERT (Transformers)
- Librosa, SpeechRecognition
- Streamlit (for UI – optional)
- VS Code (development environment)

## 📁 Project Structure

```
MindReader/
├── app/
│   ├── webcam_emotion.py
│   ├── voice_emotion.py
│   ├── text_sentiment.py
│   ├── mood_tracker.py
│   └── ...
├── templates/
│   └── index.html
├── requirements.txt
├── README.md
├── run_app.sh / run_app.bat
```

## 🛠️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/SOORYA-MK/MindReader-Emotion-Detection-and-Mental-Health-Monitoring-from-Multimodal-Inputs.git
cd MindReader-Emotion-Detection-and-Mental-Health-Monitoring-from-Multimodal-Inputs

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app/main.py
```

## 📦 Requirements

Install from:
```bash
pip install -r requirements.txt
```

## 🧠 Future Enhancements

- [ ] Streamlit-based user dashboard
- [ ] Mood history visualization
- [ ] Improved voice emotion classifier
- [ ] Real-time sentiment logging

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Author

- **Soorya M K** – [GitHub Profile](https://github.com/SOORYA-MK)
