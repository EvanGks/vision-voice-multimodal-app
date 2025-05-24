import os
import torch
import google.generativeai as genai
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from PIL import Image
import numpy as np
import uuid
from dotenv import load_dotenv, dotenv_values
from scipy.signal import resample
from kokoro import KPipeline
from huggingface_hub import hf_hub_download
import warnings
import re
import concurrent.futures
from app.backend.utils.kokoro_voices import AVAILABLE_VOICES, VOICES_BY_LANGUAGE
from app.backend.utils.text_utils import clean_text_for_tts

# Load environment variables
load_dotenv()
config = dotenv_values('.env')

# Suppress specific warnings
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")
warnings.filterwarnings("ignore", message="torch.nn.utils.weight_norm is deprecated")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
# Suppress Whisper language detection/translation warning (see https://github.com/huggingface/transformers/pull/28687)
warnings.filterwarnings("ignore", message=".*transcription using a multilingual Whisper will default to language detection.*")

def ensure_kokoro_assets(model_dir="app/backend/kokoro_assets"):
    """Download and ensure Kokoro model files are available."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "kokoro-v1_0.pth")
    config_path = os.path.join(model_dir, "config.json")
    voices_dir = os.path.join(model_dir, "voices")
    os.makedirs(voices_dir, exist_ok=True)
    
    # Download all voice files listed in AVAILABLE_VOICES
    voice_files = [f"{voice_id}.pt" for voice_id in AVAILABLE_VOICES.keys()]
    
    # Download model
    if not os.path.exists(model_path):
        print("Downloading kokoro-v1_0.pth from HuggingFace...")
        hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="kokoro-v1_0.pth", local_dir=model_dir, force_download=False)
    
    # Download config
    if not os.path.exists(config_path):
        print("Downloading config.json from HuggingFace...")
        hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="config.json", local_dir=model_dir, force_download=False)
    
    # Download all voice files into voices/voices/
    nested_voices_dir = os.path.join(voices_dir, "voices")
    os.makedirs(nested_voices_dir, exist_ok=True)
    for voice_file in voice_files:
        voice_path = os.path.join(nested_voices_dir, voice_file)
        if not os.path.exists(voice_path):
            try:
                print(f"Downloading voices/{voice_file} from HuggingFace into voices/voices/ ...")
                hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M", 
                    filename=f"voices/{voice_file}", 
                    local_dir=nested_voices_dir, 
                    repo_type="model",
                    force_download=False
                )
            except Exception as e:
                print(f"Warning: Could not download {voice_file}: {str(e)}")
    
    return model_path, config_path, voices_dir

class ModelManager:
    """
    Manager class for handling AI model interactions.
    Manages STT, TTS, and multimodal NLP models.
    """
    
    def __init__(self, upload_folder=None):
        """Initialize the model manager with all required models."""
        self.upload_folder = upload_folder if upload_folder else config.get("UPLOAD_FOLDER", "app/uploads")
        # Configure Google API
        api_key = config.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in .env file")
        genai.configure(api_key=api_key)

        # Initialize Whisper model for STT
        stt_model_name = config.get("STT_MODEL", "openai/whisper-tiny")
        self.stt_processor = WhisperProcessor.from_pretrained(stt_model_name)
        self.stt_model = WhisperForConditionalGeneration.from_pretrained(stt_model_name)

        # Ensure Kokoro model files are present (auto-download if missing)
        kokoro_model_dir = config.get("KOKORO_ASSETS_DIR", "app/backend/kokoro_assets")
        model_path, config_path, voices_dir = ensure_kokoro_assets(model_dir=kokoro_model_dir)
        # Set KOKORO_PATH env var for Kokoro to find its assets
        os.environ["KOKORO_PATH"] = os.path.abspath(kokoro_model_dir)
        # Initialize Kokoro TTS pipeline
        kokoro_repo_id = config.get("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
        self.kokoro_pipeline = KPipeline(
            lang_code='a',
            repo_id=kokoro_repo_id
        )

        # Configure Gemini model
        gemini_model_name = config.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
        self.gemini_model = genai.GenerativeModel(gemini_model_name)
        print("All models loaded successfully")

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using Whisper model.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        # Load audio
        audio_array, sampling_rate = sf.read(audio_path)
        
        # Resample audio to 16000 Hz if necessary
        if sampling_rate != 16000:
            num_samples = round(len(audio_array) * 16000 / sampling_rate)
            audio_array = resample(audio_array, num_samples)
            sampling_rate = 16000
        
        # Process audio with Whisper
        input_features = self.stt_processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt",
            # Explicitly set language to English to address the warning
            language="en"
        ).input_features

        # Pad or truncate input_features to length 3000 (required by Whisper)
        required_length = 3000
        seq_len = input_features.shape[-1]
        if seq_len > required_length:
            input_features = input_features[..., :required_length]
        elif seq_len < required_length:
            pad_width = required_length - seq_len
            input_features = torch.nn.functional.pad(input_features, (0, pad_width))
        
        # Generate transcription
        predicted_ids = self.stt_model.generate(input_features)
        transcription = self.stt_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def process_image_and_query(self, image_path, query):
        """
        Process an image and a query using Gemini model.
        
        Args:
            image_path: Path to the image file
            query: Text query about the image
            
        Returns:
            Response text from Gemini
        """
        # Open and prepare the image
        image = Image.open(image_path)
        
        # Create detailed instruction prompt
        prompt = f"""
        You are an AI assistant specialized in analyzing images and providing detailed, accurate answers about them.
        
        Please analyze the image and answer this question: {query}
        
        Guidelines:
        - Be detailed and descriptive in your explanation
        - If the answer is not apparent from the image, acknowledge the limitation
        - If there are ambiguities, mention them
        - Use a conversational, helpful tone
        - Focus on providing factual information
        - If the question asks for calculations or text extraction, perform them accurately
        - IMPORTANT: Format your response as plain text without markdown formatting
        - For emphasis, use natural language indicators like "importantly" or "note that" instead of bold or italics
        - For mathematical expressions, write them in a way that can be easily read aloud (e.g., "x squared plus 2x equals 10")
        - Avoid using special characters like asterisks, underscores, dollar signs, or backticks for formatting
        """
        
        # Prepare content for Gemini (image + text)
        response = self.gemini_model.generate_content([
            image,
            prompt
        ])
        
        return response.text
    
    def text_to_speech(self, text, voice='af_heart', speed=1.2):
        """
        Convert text to speech using Kokoro TTS with performance optimizations.
        Args:
            text: Text to convert to speech
            voice: Voice name (see Kokoro docs for options)
            speed: Speech speed (default 1.2 - slightly faster than original)
        Returns:
            Path to the generated audio file
        """
        # Skip text cleaning for simple responses without special formatting
        has_special_formatting = any(marker in text for marker in ['$', '**', '*', '__', '_', '#', '```', '`', '[', '!['])
        if has_special_formatting:
            cleaned_text = clean_text_for_tts(text)
        else:
            cleaned_text = text
        
        # Validate that the voice exists
        voices_dir = os.path.join("app", "backend", "kokoro_assets", "voices", "voices")
        local_voice_path = os.path.join(voices_dir, f"{voice}.pt")
        use_voice = voice
        if os.path.exists(local_voice_path):
            print(f"[TTS] Using local voice file: {local_voice_path}")
            use_voice = local_voice_path
        else:
            if voice not in AVAILABLE_VOICES:
                print(f"Warning: Voice '{voice}' not found in AVAILABLE_VOICES. Falling back to 'af_heart'.")
                use_voice = 'af_heart'
            else:
                print(f"[TTS] Local voice file not found for '{voice}'. Will attempt to download or use default Kokoro behavior.")
        
        # For shorter texts, process as a single chunk - increased threshold to improve reliability
        if len(cleaned_text) < 150:
            try:
                generator = self.kokoro_pipeline(cleaned_text, voice=use_voice, speed=speed)
                audio_chunks = []
                for _, _, audio in generator:
                    audio_chunks.append(audio)
                if len(audio_chunks) == 0:
                    raise RuntimeError("No audio generated by Kokoro TTS.")
                full_audio = np.concatenate(audio_chunks)
            except Exception as e:
                print(f"Error in single-chunk TTS processing: {str(e)}")
                # Fallback to non-parallel processing for troublesome text
                return self._fallback_tts(cleaned_text, use_voice, speed)
        else:
            try:
                # Create proper chunks that won't cause the RuntimeError in espeak
                # This avoids the words_mismatch.py error by using complete sentences
                sentences = []
                current = ""
                # Split by punctuation with proper regex to maintain punctuation in output
                for part in re.split(r'([.!?]+)', cleaned_text):
                    current += part
                    if re.search(r'[.!?]+$', part):
                        sentences.append(current.strip())
                        current = ""
                
                if current:  # Add any remaining text
                    sentences.append(current.strip())
                
                # Group sentences into chunks of reasonable size
                text_chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    # If adding this sentence would make the chunk too long, start a new chunk
                    if len(current_chunk) + len(sentence) > 150:  # 150 chars per chunk - smaller for reliability
                        if current_chunk:
                            text_chunks.append(current_chunk)
                        current_chunk = sentence
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                
                # Add the last chunk if it exists
                if current_chunk:
                    text_chunks.append(current_chunk)
                
                # If we have problematic small chunks, process sequentially
                if any(len(chunk) < 5 for chunk in text_chunks) or len(text_chunks) == 1:
                    return self._fallback_tts(cleaned_text, use_voice, speed)
                
                # Process chunks in parallel with error handling
                all_audio_chunks = []
                
                def process_chunk(chunk):
                    try:
                        chunk_audio_parts = []
                        generator = self.kokoro_pipeline(chunk, voice=use_voice, speed=speed)
                        for _, _, audio in generator:
                            chunk_audio_parts.append(audio)
                        if chunk_audio_parts:
                            return np.concatenate(chunk_audio_parts)
                        return np.array([])  # Empty array if no audio generated
                    except Exception as e:
                        print(f"Error processing TTS chunk: {str(e)}")
                        return np.array([])  # Return empty on error
                
                # Use ThreadPoolExecutor for parallel processing with fewer workers
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit all chunks for processing
                    future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in text_chunks}
                    
                    # Collect results as they complete and handle errors
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        try:
                            chunk_audio = future.result()
                            if len(chunk_audio) > 0:
                                all_audio_chunks.append(chunk_audio)
                        except Exception as e:
                            chunk = future_to_chunk[future]
                            print(f"Error retrieving chunk result for '{chunk}': {str(e)}")
                
                # If parallel processing failed, fall back to sequential
                if not all_audio_chunks:
                    print("Parallel processing failed, falling back to sequential processing")
                    return self._fallback_tts(cleaned_text, use_voice, speed)
                    
                # Concatenate all processed chunks
                full_audio = np.concatenate(all_audio_chunks)
            
            except Exception as e:
                print(f"Error in parallel TTS processing: {str(e)}")
                # Fallback to sequential processing
                return self._fallback_tts(cleaned_text, use_voice, speed)
        
        # Use a lower sample rate for faster processing (was 24000)
        output_sample_rate = 22050  # Still good quality but slightly faster
        
        output_filename = f"{uuid.uuid4()}_response.wav"
        output_path = os.path.join(self.upload_folder, output_filename)
        sf.write(output_path, full_audio, output_sample_rate)
        return output_filename
    
    def _fallback_tts(self, text, voice, speed):
        """
        Fallback method for text-to-speech that processes the entire text sequentially.
        Used when parallel processing fails.
        
        Args:
            text: Text to convert to speech
            voice: Voice name to use
            speed: Speech speed
            
        Returns:
            Path to the generated audio file
        """
        print(f"[TTS] Using fallback sequential processing for text of length {len(text)}")
        try:
            # Process the entire text as a single unit
            generator = self.kokoro_pipeline(text, voice=voice, speed=speed)
            audio_chunks = []
            for _, _, audio in generator:
                audio_chunks.append(audio)
            
            if not audio_chunks:
                # If still failing, try splitting by sentences and process one by one
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                        
                    try:
                        gen = self.kokoro_pipeline(sentence, voice=voice, speed=speed)
                        for _, _, audio in gen:
                            audio_chunks.append(audio)
                    except Exception as e:
                        print(f"Error processing sentence '{sentence}': {str(e)}")
                        # Continue with next sentence if one fails
            
            # If we have any audio, concatenate and return
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                output_sample_rate = 22050
                output_filename = f"{uuid.uuid4()}_response.wav"
                output_path = os.path.join(self.upload_folder, output_filename)
                sf.write(output_path, full_audio, output_sample_rate)
                return output_filename
                
            # As a last resort, generate a very simple message
            gen = self.kokoro_pipeline("I'm sorry, I couldn't generate audio for this response.", 
                                      voice=voice, speed=speed)
            error_chunks = []
            for _, _, audio in gen:
                error_chunks.append(audio)
                
            error_audio = np.concatenate(error_chunks)
            output_filename = f"{uuid.uuid4()}_error_response.wav"
            output_path = os.path.join(self.upload_folder, output_filename)
            sf.write(output_path, error_audio, 22050)
            return output_filename
            
        except Exception as e:
            print(f"Fallback TTS also failed: {str(e)}")
            # Create a simple silent audio file as last resort
            silent_audio = np.zeros(22050)  # 1 second of silence
            output_filename = f"{uuid.uuid4()}_silent_response.wav"
            output_path = os.path.join(self.upload_folder, output_filename)
            sf.write(output_path, silent_audio, 22050)
            return output_filename
        
    def get_available_voices(self):
        """
        Get list of available voices.
        
        Returns:
            Dictionary of available voices with metadata
        """
        return AVAILABLE_VOICES
        
    def get_voices_by_language(self):
        """
        Get voices grouped by language.
        
        Returns:
            Dictionary of voices grouped by language
        """
        return VOICES_BY_LANGUAGE 