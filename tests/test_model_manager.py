import os
import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from PIL import Image
from app.backend.utils.model_manager import ModelManager, ensure_kokoro_assets

class TestModelManager:
    """Test suite for the ModelManager class."""

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up mock environment variables."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        monkeypatch.setenv("UPLOAD_FOLDER", "tests/test_data")
        monkeypatch.setenv("KOKORO_ASSETS_DIR", "tests/mock_kokoro_assets")
        
    @pytest.fixture
    def mock_models(self):
        """Set up mock model responses."""
        with patch("google.generativeai.GenerativeModel") as mock_genai, \
             patch("transformers.WhisperProcessor.from_pretrained") as mock_whisper_processor, \
             patch("transformers.WhisperForConditionalGeneration.from_pretrained") as mock_whisper_model, \
             patch("app.backend.utils.model_manager.KPipeline") as mock_kokoro_pipeline, \
             patch("google.generativeai.configure") as mock_genai_configure, \
             patch("app.backend.utils.model_manager.ensure_kokoro_assets") as mock_ensure_kokoro:
            
            # Configure mocks
            mock_whisper_processor.return_value = MagicMock()
            mock_whisper_model.return_value = MagicMock()
            # Mock KPipeline as a callable that returns a callable
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.called = False
            mock_kokoro_pipeline.return_value = mock_pipeline_instance
            def side_effect(*args, **kwargs):
                mock_pipeline_instance.called = True
                return [(None, None, np.array([0.1, 0.2, 0.3]))]
            mock_pipeline_instance.side_effect = side_effect
            
            mock_genai.return_value = MagicMock()
            mock_ensure_kokoro.return_value = ("mock_model_path.pth", "mock_config.json", "mock_voices_dir")
            
            yield {
                "genai": mock_genai,
                "whisper_processor": mock_whisper_processor,
                "whisper_model": mock_whisper_model,
                "kokoro_pipeline": mock_kokoro_pipeline,
                "genai_configure": mock_genai_configure,
                "ensure_kokoro": mock_ensure_kokoro
            }
    
    def test_initialization(self, mock_env_vars, mock_models):
        """Test that the ModelManager initializes correctly."""
        model_manager = ModelManager(upload_folder="tests/test_data")
        
        # Verify that the models are loaded
        assert mock_models["whisper_processor"].called
        assert mock_models["whisper_model"].called
        assert mock_models["kokoro_pipeline"].called  # KPipeline should be called during init
        assert mock_models["genai_configure"].called
        assert mock_models["genai"].called
        assert mock_models["ensure_kokoro"].called
        assert model_manager.upload_folder == "tests/test_data"
    
    @patch("soundfile.read")
    def test_transcribe_audio(self, mock_sf_read, mock_env_vars, mock_models):
        """Test audio transcription."""
        # Setup
        mock_sf_read.return_value = (np.array([0.1, 0.2, 0.3]), 16000)
        
        model_manager = ModelManager()
        
        # Correctly prepare the Whisper processor mock to return a tensor with a shape attribute
        input_features_mock = MagicMock()
        input_features_mock.shape = [-1, 3000]
        model_manager.stt_processor.return_value.input_features = input_features_mock
        
        # Mock the Whisper model's generate and batch_decode methods
        model_manager.stt_model.generate.return_value = "mock_ids"
        model_manager.stt_processor.batch_decode.return_value = ["Transcribed text"]
        
        # Call the method
        result = model_manager.transcribe_audio("mock_audio.wav")
        
        # Assert
        assert result == "Transcribed text"
        assert mock_sf_read.called
        assert model_manager.stt_model.generate.called
        assert model_manager.stt_processor.batch_decode.called
    
    @patch("PIL.Image.open")
    def test_process_image_and_query(self, mock_image_open, mock_env_vars, mock_models):
        """Test image and query processing."""
        # Setup
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        
        model_manager = ModelManager()
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = "This is a test response."
        model_manager.gemini_model.generate_content.return_value = mock_response
        
        # Call the method
        result = model_manager.process_image_and_query("mock_image.jpg", "What's in this image?")
        
        # Assert
        assert result == "This is a test response."
        assert mock_image_open.called
        assert model_manager.gemini_model.generate_content.called
        
        # Check that generate_content was called with image and text
        call_args = model_manager.gemini_model.generate_content.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] == mock_image
        assert "What's in this image?" in call_args[1]
    
    @patch("uuid.uuid4")
    @patch("soundfile.write")
    def test_text_to_speech(self, mock_sf_write, mock_uuid, mock_env_vars, mock_models):
        """Test text to speech conversion."""
        # Setup
        mock_uuid.return_value = "test-uuid"
        
        model_manager = ModelManager()
        
        # Configure the mock Kokoro pipeline
        mock_pipeline_instance = MagicMock()
        model_manager.kokoro_pipeline = mock_pipeline_instance
        mock_pipeline_instance.return_value = [(None, None, np.array([0.1, 0.2, 0.3]))]
        
        # Call the method
        result = model_manager.text_to_speech("This is a test.", voice="af_heart", speed=1.0)
        
        # Assert
        assert result == "test-uuid_response.wav"
        assert mock_pipeline_instance.called
        assert mock_sf_write.called
    
    @patch("os.path.exists")
    def test_text_to_speech_with_special_formatting(self, mock_path_exists, mock_env_vars, mock_models):
        """Test text to speech with special formatting that needs cleaning."""
        mock_path_exists.return_value = True
        
        # Create real patch for clean_text_for_tts directly on the module
        with patch("app.backend.utils.model_manager.clean_text_for_tts") as mock_clean_text:
            mock_clean_text.return_value = "Cleaned text"
            
            model_manager = ModelManager()
            
            # Configure mock return value for the kokoro pipeline
            mock_pipeline_instance = MagicMock()
            model_manager.kokoro_pipeline = mock_pipeline_instance
            mock_pipeline_instance.return_value = [(None, None, np.array([0.1, 0.2, 0.3]))]
            
            # Call with text containing markdown-like formatting
            with patch("uuid.uuid4", return_value="test-uuid"), \
                 patch("soundfile.write") as mock_sf_write:
                
                # Pass text with special formatting that should trigger clean_text_for_tts
                text_with_formatting = "Text with **bold** and *italic*"
                result = model_manager.text_to_speech(text_with_formatting, voice="af_heart", speed=1.0)
            
            # Assert
            assert result == "test-uuid_response.wav"
            assert mock_pipeline_instance.called
            assert mock_sf_write.called
            assert mock_clean_text.called
    
    def test_get_available_voices(self, mock_env_vars, mock_models):
        """Test retrieving available voices."""
        from app.backend.utils.kokoro_voices import AVAILABLE_VOICES
        
        model_manager = ModelManager()
        voices = model_manager.get_available_voices()
        
        assert voices == AVAILABLE_VOICES
        assert isinstance(voices, dict)
        assert len(voices) > 0
    
    def test_get_voices_by_language(self, mock_env_vars, mock_models):
        """Test retrieving voices organized by language."""
        from app.backend.utils.kokoro_voices import VOICES_BY_LANGUAGE
        
        model_manager = ModelManager()
        voices_by_lang = model_manager.get_voices_by_language()
        
        assert voices_by_lang == VOICES_BY_LANGUAGE
        assert isinstance(voices_by_lang, dict)
        assert len(voices_by_lang) > 0

    def test_ensure_kokoro_assets(self):
        """Test the ensure_kokoro_assets function."""
        # Define a mock function that tracks its calls
        mock_hf_download = MagicMock(return_value="/mock/path/to/file")
        
        # Use a context manager to patch the necessary functions/objects
        with patch("os.makedirs") as mock_makedirs, \
             patch("os.path.exists") as mock_path_exists, \
             patch("app.backend.utils.model_manager.AVAILABLE_VOICES", {"test_voice": {}}), \
             patch("app.backend.utils.model_manager.hf_hub_download", mock_hf_download):
            
            # Mock path existence checks - all paths should return False to trigger downloads
            mock_path_exists.return_value = False
            
            # Call the function
            model_path, config_path, voices_dir = ensure_kokoro_assets("test/assets/dir")
            
            # Verify directories were created
            assert mock_makedirs.call_count >= 3  # At least for main dir, voices dir, and nested voices dir
            
            # Verify downloads were initiated - ensure it was called at least once
            assert mock_hf_download.call_count > 0
            
            # Verify specific calls to download model and config files
            kokoro_calls = [call for call in mock_hf_download.call_args_list 
                            if 'kokoro-v1_0.pth' in str(call) or 'config.json' in str(call)]
            assert len(kokoro_calls) >= 2  # At least model and config file 