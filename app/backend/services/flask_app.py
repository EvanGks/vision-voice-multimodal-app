import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
from app.backend.utils.model_manager import ModelManager
from dotenv import dotenv_values

app = Flask(__name__)
CORS(app)

# Load environment variables (should be done with python-dotenv in production)
config = dotenv_values('.env')
app.config['SECRET_KEY'] = config.get('FLASK_SECRET_KEY', 'default-dev-key')
app.config['UPLOAD_FOLDER'] = config.get('UPLOAD_FOLDER', 'app/uploads')
app.config['MAX_CONTENT_LENGTH'] = int(config.get('MAX_CONTENT_LENGTH', 16 * 1000 * 1000))

# Initialize model manager
model_manager = ModelManager(upload_folder=app.config['UPLOAD_FOLDER'])

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio file to text."""
    try:
        # Check if audio file is present in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty audio filename'}), 400
        
        # Save file with unique name
        audio_filename = secure_filename(f"{uuid.uuid4()}_{audio_file.filename}")
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        audio_file.save(audio_path)
        
        # Process the voice input (transcribe)
        transcription = model_manager.transcribe_audio(audio_path)
        
        return jsonify({
            'transcription': transcription,
            'audio_path': audio_path
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during transcription.', 'details': str(e)}), 500

@app.route('/api/generate_response', methods=['POST'])
def generate_response():
    """Generate text response using the model."""
    try:
        data = request.form
        image_path = data.get('image_path')
        query = data.get('query')
                       
        # Validate inputs
        if not image_path:
            return jsonify({'error': 'No image provided'}), 400
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Process the image and query with Gemini
        response_text = model_manager.process_image_and_query(image_path, query)
        
        return jsonify({
            'response': response_text
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An error occurred while generating the response.', 'details': str(e)}), 500

@app.route('/api/text_to_speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech."""
    try:
        data = request.json
        text = data.get('text')
        voice = data.get('voice', 'af_heart')  # Default voice
        speed = float(data.get('speed', 1.0))  # Default speed
        high_performance = data.get('high_performance', False)  # Default to normal mode
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # If high_performance mode is enabled, increase speed
        if high_performance:
            speed = max(speed, 1.2)  # Ensure minimum speed of 1.2 in high-performance mode
        
        # Convert response to speech
        audio_response_path = model_manager.text_to_speech(text, voice=voice, speed=speed)
        
        # Check that the file exists and is not empty
        audio_full_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_response_path)
        if not os.path.exists(audio_full_path) or os.path.getsize(audio_full_path) == 0:
            return jsonify({'error': 'Audio generation failed. The audio file is missing or empty.'}), 500
        
        return jsonify({
            'audio_response': audio_response_path
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during text-to-speech conversion.', 'details': str(e)}), 500

@app.route('/api/voices', methods=['GET'])
def get_voices():
    """Get available voices for TTS."""
    try:
        voices = model_manager.get_available_voices()
        return jsonify(voices), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An error occurred while retrieving voices.', 'details': str(e)}), 500

@app.route('/api/voices_by_language', methods=['GET'])
def get_voices_by_language():
    """Get available voices organized by language."""
    try:
        voices_by_language = model_manager.get_voices_by_language()
        voices_data = model_manager.get_available_voices()
        
        # Build a structured response with full voice information
        response = {}
        for language, voice_ids in voices_by_language.items():
            voices_info = []
            for voice_id in voice_ids:
                if voice_id in voices_data:
                    voice_info = voices_data[voice_id].copy()
                    voice_info['id'] = voice_id
                    voices_info.append(voice_info)
            response[language] = voices_info
            
        return jsonify(response), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An error occurred while retrieving voices by language.', 'details': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_query():
    """Process an image and voice query from the user."""
    try:
        # Check if an image file is present in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty image filename'}), 400
        
        # Check if audio file is present in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty audio filename'}), 400
        
        # Get optional parameters
        high_performance = request.form.get('high_performance', 'false').lower() == 'true'
        voice = request.form.get('voice', 'af_heart')  # Default voice
        speed = float(request.form.get('speed', 1.0))  # Default speed
        
        # If high_performance mode is enabled, increase speed
        if high_performance:
            speed = max(speed, 1.2)  # Ensure minimum speed of 1.2 in high-performance mode
        
        # Save files with unique names
        image_filename = secure_filename(f"{uuid.uuid4()}_{image_file.filename}")
        audio_filename = secure_filename(f"{uuid.uuid4()}_{audio_file.filename}")
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        
        image_file.save(image_path)
        audio_file.save(audio_path)
        
        # Process the voice input (transcribe)
        transcription = model_manager.transcribe_audio(audio_path)
        
        # Process the image and transcribed query with Gemini
        response_text = model_manager.process_image_and_query(image_path, transcription)
        
        # Convert response to speech
        audio_response_path = model_manager.text_to_speech(response_text, voice=voice, speed=speed)
        
        # Return the results
        return jsonify({
            'transcription': transcription,
            'response': response_text,
            'audio_response': audio_response_path
        }), 200
        
    except Exception as e:
        # Log the error for debugging
        import traceback
        traceback.print_exc()
        # Return a user-friendly error message and the exception string for debugging
        return jsonify({'error': 'An internal error occurred. Please try again later.', 'details': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True) 