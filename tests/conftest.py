import os
import pytest
import shutil
from pathlib import Path
import numpy as np

@pytest.fixture(scope="session")
def test_dir():
    """Create and return the test directory path."""
    test_dir = Path("tests/test_data")
    test_dir.mkdir(exist_ok=True)
    return test_dir

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_files(test_dir):
    """Clean up test files after all tests have run."""
    yield
    # Clean up test data files but keep the directory
    for item in test_dir.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

@pytest.fixture
def sample_image(test_dir):
    """Create a simple test image."""
    from PIL import Image
    
    # Create a small test image
    img_path = test_dir / "test_image.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)
    
    yield str(img_path)
    
    # Cleanup happens in cleanup_test_files

@pytest.fixture
def sample_audio(test_dir):
    """Create a simple test audio file with proper WAV format."""
    import soundfile as sf
    import wave
    
    # Create a simple sine wave audio file with standard WAV format
    audio_path = test_dir / "test_audio.wav"
    
    # Create 1 second of audio at 16kHz
    samplerate = 16000
    duration = 1  # seconds
    t = np.linspace(0, duration, int(samplerate * duration))
    data = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Ensure the data is in float32 format
    data = data.astype(np.float32)
    
    # Write using soundfile which handles the format correctly
    sf.write(str(audio_path), data, samplerate, format='WAV', subtype='PCM_16')
    
    # Verify the file exists and has content
    assert os.path.exists(audio_path)
    assert os.path.getsize(audio_path) > 0
    
    # Try opening it with wave to verify it's valid
    with wave.open(str(audio_path), 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == samplerate
    
    yield str(audio_path)
    
    # Cleanup happens in cleanup_test_files

@pytest.fixture
def mock_kokoro_assets_dir(test_dir):
    """Create and setup a mock Kokoro assets directory."""
    kokoro_dir = test_dir / "mock_kokoro_assets"
    voices_dir = kokoro_dir / "voices" / "voices"
    
    # Create all necessary directories
    kokoro_dir.mkdir(exist_ok=True)
    voices_dir.mkdir(parents=True, exist_ok=True)
    
    # Create empty placeholder files to simulate downloaded assets
    (kokoro_dir / "kokoro-v1_0.pth").touch()
    (kokoro_dir / "config.json").touch()
    (voices_dir / "af_heart.pt").touch()
    
    # Set environment variable for tests
    os.environ["KOKORO_ASSETS_DIR"] = str(kokoro_dir)
    os.environ["KOKORO_PATH"] = str(kokoro_dir)
    
    yield str(kokoro_dir)
    
    # Cleanup happens in cleanup_test_files 