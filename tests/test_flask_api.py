import os
import pytest
import json
from unittest.mock import patch, MagicMock
from io import BytesIO
from app.backend.services.flask_app import app
from app.backend.utils.model_manager import ModelManager

class TestFlaskAPI:
    """Test suite for the Flask API."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the Flask app."""
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = 'tests/test_data'
        
        # Ensure test upload folder exists
        os.makedirs('tests/test_data', exist_ok=True)
        
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def mock_model_manager(self, monkeypatch):
        """Create a mock ModelManager for testing."""
        mock_manager = MagicMock(spec=ModelManager)
        mock_manager.transcribe_audio.return_value = "What is in this image?"
        mock_manager.process_image_and_query.return_value = "The image contains a cat."
        mock_manager.text_to_speech.return_value = "response_audio.wav"
        mock_manager.get_available_voices.return_value = {"af_heart": {"name": "Heart", "gender": "Female"}}
        mock_manager.get_voices_by_language.return_value = {"English (African)": ["af_heart"]}
        
        # Monkeypatch the ModelManager in the flask_app module
        monkeypatch.setattr('app.backend.services.flask_app.model_manager', mock_manager)
        
        yield mock_manager
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        assert response.json['status'] == 'ok'
    
    def test_transcribe_audio(self, client, mock_model_manager):
        """Test the transcribe endpoint."""
        # Create test data
        data = {
            'audio': (BytesIO(b'fake audio data'), 'test.wav')
        }
        
        response = client.post('/api/transcribe', data=data)
        
        assert response.status_code == 200
        assert response.json['transcription'] == "What is in this image?"
        assert 'audio_path' in response.json
        assert mock_model_manager.transcribe_audio.called
    
    def test_generate_response(self, client, mock_model_manager):
        """Test the generate_response endpoint."""
        # Create test data
        data = {
            'image_path': 'test/path/image.jpg',
            'query': 'What is in this image?'
        }
        
        response = client.post('/api/generate_response', data=data)
        
        assert response.status_code == 200
        assert response.json['response'] == "The image contains a cat."
        assert mock_model_manager.process_image_and_query.called
    
    def test_text_to_speech(self, client, mock_model_manager):
        """Test the text_to_speech endpoint."""
        # Create test data
        json_data = {
            'text': 'The image contains a cat.',
            'voice': 'af_heart',
            'speed': 1.0
        }
        
        # Also mock os.path.exists to return True for the audio file check
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024):  # Non-zero file size
            response = client.post('/api/text_to_speech', json=json_data)
        
        assert response.status_code == 200
        assert response.json['audio_response'] == "response_audio.wav"
        assert mock_model_manager.text_to_speech.called
    
    def test_get_voices(self, client, mock_model_manager):
        """Test the get_voices endpoint."""
        response = client.get('/api/voices')
        
        assert response.status_code == 200
        assert "af_heart" in response.json
        assert mock_model_manager.get_available_voices.called
    
    def test_get_voices_by_language(self, client, mock_model_manager):
        """Test the get_voices_by_language endpoint."""
        # Set up mock to return in the expected format for the endpoint
        mock_model_manager.get_voices_by_language.return_value = {"English (African)": ["af_heart"]}
        mock_model_manager.get_available_voices.return_value = {
            "af_heart": {"name": "Heart", "gender": "Female"}
        }
        
        response = client.get('/api/voices_by_language')
        
        assert response.status_code == 200
        assert "English (African)" in response.json
        assert mock_model_manager.get_voices_by_language.called
        assert mock_model_manager.get_available_voices.called
    
    def test_process_missing_image(self, client, mock_model_manager):
        """Test process endpoint with missing image."""
        # Create test data
        data = {}
        
        response = client.post('/api/process', data=data)
        
        assert response.status_code == 400
        assert 'No image file provided' in response.json['error']
    
    def test_process_missing_audio(self, client, mock_model_manager):
        """Test process endpoint with missing audio."""
        # Create test data with image but no audio
        data = {
            'image': (BytesIO(b'fake image data'), 'test.jpg')
        }
        
        response = client.post('/api/process', data=data)
        
        assert response.status_code == 400
        assert 'No audio file provided' in response.json['error']
    
    def test_process_successful(self, client, mock_model_manager):
        """Test successful processing of image and audio."""
        # Create test data
        data = {
            'image': (BytesIO(b'fake image data'), 'test.jpg'),
            'audio': (BytesIO(b'fake audio data'), 'test.wav'),
            'voice': 'af_heart',
            'speed': '1.0',
            'high_performance': 'false'
        }
        
        response = client.post('/api/process', data=data)
        
        # Check response
        assert response.status_code == 200
        assert response.json['transcription'] == "What is in this image?"
        assert response.json['response'] == "The image contains a cat."
        assert response.json['audio_response'] == "response_audio.wav"
        
        # Verify model manager method calls
        assert mock_model_manager.transcribe_audio.called
        assert mock_model_manager.process_image_and_query.called
        assert mock_model_manager.text_to_speech.called
    
    def test_process_exception_handling(self, client, mock_model_manager):
        """Test exception handling in the process endpoint."""
        # Configure mock to raise an exception
        mock_model_manager.transcribe_audio.side_effect = Exception("Test error")
        
        # Create test data
        data = {
            'image': (BytesIO(b'fake image data'), 'test.jpg'),
            'audio': (BytesIO(b'fake audio data'), 'test.wav')
        }
        
        response = client.post('/api/process', data=data)
        
        # Check response
        assert response.status_code == 500
        assert 'Test error' in response.json['details'] 