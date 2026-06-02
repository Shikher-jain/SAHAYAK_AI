from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from backend.ingestion.text import chunk_text


def _is_private_host(hostname: str) -> bool:
    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True

    for info in infos:
        candidate = info[4][0]
        try:
            ip = ipaddress.ip_address(candidate)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return True
        except ValueError:
            return True
    return False


def validate_public_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL must include a valid host")

    host = parsed.hostname.lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Localhost URLs are not allowed")
    if host.endswith(".local"):
        raise ValueError("Local network hostnames are not allowed")
    if _is_private_host(host):
        raise ValueError("Private or internal network addresses are not allowed")
    return parsed.geturl()


def fetch_url_text(url: str) -> str:
    safe_url = validate_public_url(url)
    response = requests.get(
        safe_url,
        timeout=30,
        headers={"User-Agent": "SahayakAI/1.0 (+https://example.local)"},
        allow_redirects=True,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def chunk_url(url: str):
    text = fetch_url_text(url)
    return chunk_text(text, strategy="semantic")
