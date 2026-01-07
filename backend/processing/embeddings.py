from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    return model.encode(text)
    return model.encode(text)
# backend/processing/embeddings.py

from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class EmbeddingProcessor:
    def __init__(self):
        # Text embedding model
        self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # Image embedding model (CLIP)
        self.image_model = SentenceTransformer('clip-ViT-B-32')
        # Audio transcription model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)

    def embed_text(self, text: str):
        return self.text_model.encode(text).tolist()

    def embed_image(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        return self.image_model.encode([image])[0].tolist()

    def transcribe_audio(self, audio_path: str):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)
        inputs = self.audio_processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values.to(self.device)
        with torch.no_grad():
            logits = self.audio_model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.audio_processor.batch_decode(predicted_ids)[0]
        return transcription

    def embed_audio(self, audio_path: str):
        """Transcribe audio and return embedding for transcription"""
        text = self.transcribe_audio(audio_path)
        return self.embed_text(text), text
