# 🧠 Multimodal AI Visual Assistant

A full-stack, production-grade multimodal AI assistant that enables users to interact with images using natural language and voice. Get instant, multilingual spoken answers powered by state-of-the-art AI models.

## 🚀 Features

- **Image Upload & Analysis**: Ask questions about any image using your voice.
- **Voice Input & Transcription**: Record queries, transcribed in real-time with OpenAI Whisper.
- **Advanced Multimodal Reasoning**: Google Gemini 2.5 Flash analyzes images and queries for detailed, factual answers.
- **Multilingual Voice Output**: Hear responses in your chosen language, accent, and gender using Kokoro TTS.
- **Voice Customization**: Select from dozens of voices (male/female), languages, and adjust speech rate.
- **Accessible, Responsive UI**: Gradio interface designed for keyboard navigation, screen readers, and all devices.
- **Automated Asset Management**: All TTS models and voices are auto-downloaded on first run.
- **Robust Error Handling**: User-friendly error messages and backend logging.
- **Comprehensive Testing**: Pytest suite covers backend, frontend, and model logic.

## 📸 Screenshots

Below are some screenshots of the application in action. (Add your images to the `assets/` directory and reference them here.)

![Main UI](assets/UI_demo.png)
![Example](assets/example_demo.png)

## 🏗️ Architecture

```
User
  │
  ▼
Gradio UI (app/frontend/gradio_app.py)
  │
  ▼
Flask API (app/backend/services/flask_app.py)
  │
  ▼
ModelManager (app/backend/utils/model_manager.py)
  ├─ Whisper (Speech-to-Text)
  ├─ Gemini (Image+Text Reasoning)
  └─ Kokoro TTS (Text-to-Speech)
```

- **Concurrent Backend & Frontend**: Both run together via `main.py` (threaded).
- **All assets and models are managed automatically.**

## 🌍 Supported Languages & Voices

- **English** (American, British,): Multiple male/female voices
- **Japanese**: Male/female voices
- **Mandarin Chinese**: Multiple voices
- **French**: Female voice
- **Spanish**: Male/female voices
- **Italian**: Male/female voices
- **Brazilian Portuguese**: Male/female voices
- **Hindi**: Male/female voices

See [`app/backend/utils/kokoro_voices.py`](app/backend/utils/kokoro_voices.py) for the full list.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/EvanGks/vision-voice-multimodal-app.git
   cd multimodal-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in your Google API key and Flask secret.
   - **All configuration variables (API keys, model names, upload folders, etc.) are loaded exclusively from the `.env` file in the root directory.**
   - The application does **not** read configuration from system environment variables—**all configuration must be set in `.env`**.
   - All model assets will be auto-downloaded.

5. **Run the application:**
   ```bash
   python main.py
   ```
   - Access the UI at [http://localhost:7860](http://localhost:7860)

## 🗂️ Project Structure

```
Multimodal_AI/
├── app/
│   ├── backend/
│   │   ├── services/           # Flask API endpoints
│   │   ├── utils/              # Model management, voice metadata, text utils
│   │   └── kokoro_assets/      # TTS model assets (auto-downloaded)
│   ├── frontend/               # Gradio UI implementation
│   └── uploads/                # Uploaded files (auto-cleaned)
├── assets/                     # Screenshots and static assets
├── tests/                      # Test suite (API, UI, models)
├── .env, .env.example          # Environment variables
├── .gitignore                  # Git ignore file
├── LICENSE                     # MIT License
├── main.py                     # Application entry point
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
```

- All code is modular and organized for clarity and extensibility.
- The `assets/` directory is the recommended place for screenshots and static files.

## 🧪 Testing

Run all tests with:
```bash
pytest
```
- Tests cover API endpoints, UI workflow, and model logic.
- Test data and assets are auto-managed and cleaned up.

## ♿ Accessibility & UX

- **Keyboard navigation** and **screen reader** support.
- **High color contrast** and **responsive design**.
- **Clear feedback** for all user actions and errors.
- **Semantic HTML** and ARIA attributes for assistive technologies.
- **Resizable text** and mobile-friendly layout.

## 🔒 Security

- **No secrets in code**—all credentials via `.env`.
- **All configuration is managed via the `.env` file only. System environment variables are not used for configuration.**
- **Strict file upload validation** and privacy cleanup.
- **Sensitive/model files are git-ignored.**
- **API key management** and environment-based configuration.

## 🧩 Extensibility

- Add new voices/languages by updating `kokoro_voices.py`.
- Swap or extend models via `ModelManager` and configure them in `.env`.
- **All configuration is centralized in `.env` for easy reproducibility and sharing.**
- Modular, testable codebase for rapid prototyping.

## ✨ Why This Project Stands Out

- **End-to-end AI workflow**: From voice to vision to speech, all in one app.
- **Production best practices**: Security, error handling, accessibility, and testing.
- **Portfolio-ready**: Demonstrates full-stack AI, modern Python, and real-world deployment skills.
- **Comprehensive documentation and code comments** for maintainability.
- **Automated asset management** for seamless setup.

## 🚀 Future Enhancements

- Add support for additional languages and regional accents in TTS and STT.
- Implement real-time streaming responses for faster feedback.
- Develop a mobile-friendly or native mobile UI.
- Integrate more advanced image analysis models (e.g., OCR, object detection).
- Add user authentication and personalized settings.
- Enable cloud deployment with scalable infrastructure.
- Provide downloadable audio/text transcripts for user queries.
- Add progress indicators and better feedback for long-running operations.
- Expand accessibility features (e.g., high-contrast mode, localization).

## 📄 License

This project is licensed under the MIT License (c) 2025 Evan GKS. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://huggingface.co/openai/whisper-tiny)
- [Google Gemini](https://ai.google.dev/)
- [Kokoro TTS](https://github.com/hexgrad/Kokoro)
- [Gradio](https://gradio.app/)
- [Flask](https://flask.palletsprojects.com/)

## 📬 Contact
For questions or feedback, please reach out via:

- **GitHub:** [EvanGks](https://github.com/EvanGks)
- **X (Twitter):** [@Evan6471133782](https://x.com/Evan6471133782)
- **LinkedIn:** [Evangelos Gakias](https://www.linkedin.com/in/evangelos-gakias-346a9072)
- **Kaggle:** [evangelosgakias](https://www.kaggle.com/evangelosgakias)
- **Email:** [evangks88@gmail.com](mailto:evangks88@gmail.com)

---
