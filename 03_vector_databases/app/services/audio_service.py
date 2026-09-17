import os
import shutil
from transformers import pipeline

print("Loading Whisper model for audio transcription...")
# Using the 'tiny' model so it downloads fast and doesn't melt your RAM
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

def process_audio_file(upload_file) -> str:
    """
    Saves the uploaded FastAPI file temporarily, transcribes it to text,
    and cleans up the temporary file.
    """
    temp_path = f"temp_{upload_file.filename}"
    
    # Save the uploaded file to disk temporarily
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        
    try:
        # Transcribe the audio
        print(f"Transcribing {temp_path}...")
        result = transcriber(temp_path)
        transcribed_text = result["text"].strip()
        return transcribed_text
    finally:
        # Always clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)