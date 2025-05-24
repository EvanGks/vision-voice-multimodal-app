import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show TensorFlow warnings and errors
import threading
import atexit
from app.backend.services.flask_app import app as flask_app
from app.frontend.gradio_app import launch_gradio

UPLOADS_DIR = os.path.join("app", "uploads")

def cleanup_uploads_folder():
    """Clean the uploads folder by deleting all files within it."""
    if os.path.exists(UPLOADS_DIR):
        for filename in os.listdir(UPLOADS_DIR):
            file_path = os.path.join(UPLOADS_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")

# Clean up uploads on startup
cleanup_uploads_folder()

atexit.register(cleanup_uploads_folder)

def run_flask():
    """Run the Flask backend server."""
    flask_app.run(debug=False, port=5000, use_reloader=False)

def run_gradio():
    """Launch the Gradio frontend interface."""
    launch_gradio()

if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # Start Flask backend in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Start Gradio frontend (main thread)
    run_gradio() 