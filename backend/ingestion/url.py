import requests
from bs4 import BeautifulSoup

from backend.ingestion.text import chunk_text


def fetch_url_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def chunk_url(url: str):
    text = fetch_url_text(url)
    return chunk_text(text)
