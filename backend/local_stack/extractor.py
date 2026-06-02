import youtube_transcript_api

from backend.ingestion.image import ocr_image_bytes
from backend.ingestion.pdf import extract_pdf_text_from_bytes
from backend.ingestion.url import fetch_url_text

def extract_pdf(file_bytes):
    return extract_pdf_text_from_bytes(file_bytes)

def extract_image(file_bytes):
    return ocr_image_bytes(file_bytes)
    
def extract_url(url):
    return fetch_url_text(url)

def extract_youtube(video_id):
    transcript = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(video_id)
    return "\n".join([x["text"] for x in transcript])
