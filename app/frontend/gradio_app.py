import os
import gradio as gr
import requests
from dotenv import dotenv_values

def launch_gradio():
    """Create and launch the Gradio interface."""
    
    # Define the backend API URLs
    TRANSCRIBE_API_URL = "http://localhost:5000/api/transcribe"
    RESPONSE_API_URL = "http://localhost:5000/api/generate_response"
    TTS_API_URL = "http://localhost:5000/api/text_to_speech"
    VOICES_API_URL = "http://localhost:5000/api/voices_by_language"
    
    # Global variables to store paths
    current_image_path = None
    current_audio_path = None
    
    # Get available voices from the backend
    try:
        response = requests.get(VOICES_API_URL)
        if response.status_code == 200:
            voices_by_language = response.json()
            # Get list of language options
            languages = list(voices_by_language.keys())
            # Create a dictionary to store voices for each language
            voices_options = {}
            for lang, voices in voices_by_language.items():
                # Format as {display_name: voice_id}
                voices_options[lang] = {f"{v['name']} ({v['gender']})": v['id'] for v in voices}
        else:
            # Fallback to default if API fails
            languages = ["English (African)"]
            voices_options = {
                "English (African)": {"Heart (Female)": "af_heart"}
            }
    except Exception as e:
        print(f"Error fetching voices: {str(e)}")
        languages = ["English (African)"]
        voices_options = {
            "English (African)": {"Heart (Female)": "af_heart"}
        }
    
    config = dotenv_values('.env')
    
    def transcribe_audio_query(audio):
        """
        Send audio for transcription.
        
        Args:
            audio: Audio recording of the user's query
            
        Returns:
            Transcribed text
        """
        global current_audio_path
        
        if audio is None:
            return "Please record your question."
            
        # Prepare files for the request
        files = {
            'audio': ('audio.wav', open(audio, 'rb'), 'audio/wav')
        }
        
        try:
            # Send request to the backend
            response = requests.post(TRANSCRIBE_API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                # Get transcription
                transcription = result.get('transcription', 'No transcription available')
                # Store audio path for future use
                current_audio_path = result.get('audio_path')
                
                return transcription
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                return error_msg
                
        except Exception as e:
            error_msg = f"Error communicating with backend: {str(e)}"
            return error_msg
    
    def generate_model_response(image, transcription):
        """
        Generate a response from the model.
        
        Args:
            image: Image path
            transcription: Transcribed query
            
        Returns:
            Model's text response
        """
        global current_image_path
        
        if image is None:
            return "Please upload an image first."
            
        if not transcription or transcription == "Please record your question.":
            return "Please record your question first."
        
        current_image_path = image
        
        # Prepare form data
        form_data = {
            'image_path': image,
            'query': transcription
        }
        
        try:
            # Send request to the backend
            response = requests.post(RESPONSE_API_URL, data=form_data)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', 'No response available')
                return response_text
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                return error_msg
                
        except Exception as e:
            error_msg = f"Error communicating with backend: {str(e)}"
            return error_msg
    
    def update_voice_options(language):
        """
        Update the voice options based on selected language.
        
        Args:
            language: Selected language
            
        Returns:
            Dictionary of voice options and default voice
        """
        # Get voices for selected language
        if language in voices_options:
            voices = voices_options[language]
            # Get first voice as default
            default_voice = list(voices.keys())[0] if voices else None
            return gr.update(choices=list(voices.keys()), value=default_voice)
        else:
            return gr.update(choices=[], value=None)
    
    def generate_audio_response(response_text, language, voice_selection, speed):
        """
        Generate audio response from text.
        
        Args:
            response_text: Text to convert to speech
            language: Selected language
            voice_selection: Selected voice name from dropdown
            speed: Speech speed
            
        Returns:
            Path to audio file
        """
        if not response_text or response_text.startswith("Error") or response_text.startswith("Please"):
            return None
        
        # Get voice ID based on language and voice selection
        voice_id = voices_options.get(language, {}).get(voice_selection, "af_heart")
            
        # Prepare JSON data
        json_data = {
            'text': response_text,
            'voice': voice_id,
            'speed': speed
        }
        
        try:
            # Send request to the backend
            response = requests.post(TTS_API_URL, json=json_data)
            
            if response.status_code == 200:
                result = response.json()
                audio_file = result.get('audio_response')
                
                # Prepare audio path
                audio_response_path = os.path.join(config.get("UPLOAD_FOLDER", "app/uploads"), audio_file) if audio_file else None
                
                return audio_response_path
            else:
                return None
                
        except Exception as e:
            return None
    
    # Create Gradio interface
    with gr.Blocks(title="🧠 Multimodal AI Visual Assistant", theme=gr.themes.Base()) as iface:
        gr.Markdown(
            "<h1 style='text-align: center; margin-bottom: 0.5em;'>🧠 Multimodal AI Visual Assistant</h1>",
            elem_id="app-title"
        )
        gr.Markdown(
            "<div style='text-align: center; font-size: 1.35em; font-weight: 500; margin-bottom: 1.5em;'>"
            "Interact with images using natural language: upload an image, ask questions with your voice, and receive intelligent spoken responses in multiple languages!"
            "</div>",
            elem_id="app-subtitle"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                image_input = gr.Image(
                    label="1. Upload an image",
                    type="filepath"
                )
                
                audio_input = gr.Audio(
                    label="2. Record your question",
                    type="filepath",
                    sources=["microphone"]
                )
                
                transcription_output = gr.Textbox(label="Your Question (Transcribed)")
                
                process_btn = gr.Button("3. Process", variant="primary")
                
            with gr.Column(scale=1):
                # Output components
                response_output = gr.Textbox(label="AI Response")
                
                with gr.Group():
                    gr.Markdown("### Voice Options")
                    with gr.Row():
                        language_dropdown = gr.Dropdown(
                            choices=languages,
                            value=languages[0] if languages else None,
                            label="Language",
                            info="Select language for voice output"
                        )
                    
                    with gr.Row():
                        voice_dropdown = gr.Dropdown(
                            choices=list(voices_options.get(languages[0], {}).keys()) if languages else [],
                            value=list(voices_options.get(languages[0], {}).keys())[0] if languages and voices_options.get(languages[0]) else None,
                            label="Voice",
                            info="Choose a specific voice"
                        )
                    
                    with gr.Row():
                        speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Speech Speed",
                            info="Adjust the speaking rate (0.5 = slower, 2.0 = faster)"
                        )
                
                audio_btn = gr.Button("4. Generate Audio Response", variant="secondary")
                audio_output = gr.Audio(label="Voice Response")
        
        # Set up the event handlers
        # Auto-transcribe when audio recording is complete
        audio_input.stop_recording(
            fn=transcribe_audio_query,
            inputs=[audio_input],
            outputs=[transcription_output]
        )
        
        process_btn.click(
            fn=generate_model_response,
            inputs=[image_input, transcription_output],
            outputs=[response_output],
            show_progress=True  # Show loading indicator on the button
        )
        
        # Update voice options when language changes
        language_dropdown.change(
            fn=update_voice_options,
            inputs=[language_dropdown],
            outputs=[voice_dropdown]
        )
        
        audio_btn.click(
            fn=generate_audio_response,
            inputs=[response_output, language_dropdown, voice_dropdown, speed_slider],
            outputs=[audio_output]
        )
        
        # How It Works and Model Details side by side
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## How It Works")
                gr.Markdown("""
                1. **Upload an image** you want to ask about
                2. **Record your question** by clicking the microphone button and speaking
                3. Your question will be **automatically transcribed** when you stop recording
                4. Click **Process** to analyze the image with your question
                5. View the AI's text response
                6. **Select a language and voice** for the audio response
                7. Click **Generate Audio Response** to hear the answer in your chosen voice
                """)
            with gr.Column(scale=1):
                gr.Markdown("""
                This application combines multiple AI technologies and state-of-the-art models to create a seamless multimodal experience:
                - **Speech Recognition (STT)**: Automatic transcription of your voice questions using the [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) model.
                - **Image Understanding & OCR**: Advanced analysis of visual content and text extraction powered by [gemini-2.5-flash-preview-05-20](https://ai.google.dev/models/gemini).
                - **Natural Language Processing (NLP)**: Intelligent responses to your queries generated by [gemini-2.5-flash-preview-05-20](https://ai.google.dev/models/gemini).
                - **Multilingual Text-to-Speech (TTS)**: High-quality voice synthesis in multiple languages and voices using [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).
                """)
    
    # Launch the interface
    iface.launch(share=False)
    
    return iface

if __name__ == "__main__":
    launch_gradio() 