"""Lấy NỘI DUNG trang nguồn hoặc PDF để đưa vào RAG phiên làm việc.

Bảo mật: chỉ cho phép hostname thuộc danh sách, chặn redirect, giới hạn byte/time.
Không lưu file lên server và không giả vờ tự xác thực được dữ kiện trong PDF.
"""
from __future__ import annotations

import io
import re
from urllib.parse import urlparse

from .engine import extract_mentions, MAX_EXTRA_TEXT

MAX_UPLOAD = 3 * 1024 * 1024
ALLOWED_DOMAINS = ("who.int", "niddk.nih.gov", "cdc.gov", "vinmec.com")


def valid_domain(url: str) -> bool:
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower().rstrip(".")
        return p.scheme == "https" and p.username is None and p.password is None and p.port in (None, 443) and any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)
    except ValueError:
        return False


def _collapse(s):
    return re.sub(r"\s+", " ", s or "").strip()[:MAX_EXTRA_TEXT]


def ingest_pdf(filebytes: bytes, title: str = "PDF tải lên") -> dict:
    if not filebytes.startswith(b"%PDF-"):
        raise ValueError("Không phải file PDF hợp lệ.")
    if len(filebytes) > MAX_UPLOAD:
        raise ValueError("PDF vượt giới hạn 3 MB.")
    try:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(filebytes), strict=False)
        if len(r.pages) > 50:
            raise ValueError("PDF dài quá 50 trang. Chỉ tải tài liệu ngắn để demo.")
        text = _collapse("\n".join((p.extract_text() or "") for p in list(r.pages)[:20]))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Không đọc được PDF dạng text. PDF scan cần OCR riêng.") from exc
    if len(text) < 40:
        raise ValueError("PDF quá ít text. Có thể là PDF scan; bản demo chưa có OCR.")
    return {"title": title[:160], "url": "", "text": text, "kind": "pdf",
            "mentions": extract_mentions(text), "review": "Nguồn người dùng upload, chưa thẩm định; chỉ trích entity mention."}


def ingest_url(url: str) -> dict:
    if not valid_domain(url):
        raise ValueError("URL phải dùng HTTPS và thuộc WHO, NIDDK, CDC hoặc Vinmec để tránh SSRF.")
    try:
        import requests
        from bs4 import BeautifulSoup
        # Không cho tự follow redirect để tránh redirect sang host nội bộ.
        response = requests.get(url, timeout=(4, 10), allow_redirects=False,
                                stream=True, headers={"User-Agent": "DiabetesMAS-KB-demo/1.0"})
        if response.status_code in (301, 302, 303, 307, 308):
            raise ValueError("URL chuyển hướng. Hãy dùng địa chỉ trang đích trực tiếp thuộc danh sách nguồn cho phép.")
        response.raise_for_status()
        typ = response.headers.get("Content-Type", "").lower()
        if not ("text/html" in typ or "application/pdf" in typ):
            raise ValueError("Chỉ hỗ trợ HTML hoặc PDF từ các tên miền cho phép.")
        data = bytearray()
        for chunk in response.iter_content(chunk_size=4096):
            data.extend(chunk)
            if len(data) > MAX_UPLOAD:
                raise ValueError("Trang quá lớn (giới hạn 3 MB).")
        if "pdf" in typ:
            doc = ingest_pdf(bytes(data), title=url.split('/')[-1] or "PDF từ web")
            doc["url"] = url
            doc["kind"] = "pdf-url"
            return doc
        soup = BeautifulSoup(bytes(data), "html.parser")
        title = _collapse((soup.title.get_text(" ") if soup.title else "Nguồn web"))[:160]
        for n in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
            n.decompose()
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = _collapse(main.get_text(" ", strip=True))
        if len(text) < 40:
            raise ValueError("Trang không có đủ văn bản có thể đọc.")
        return {"title": title, "url": url, "text": text, "kind": "web",
                "mentions": extract_mentions(text), "review": "Văn bản mới tải; thực thể được phát hiện theo từ điển, không phải quan hệ bệnh học đã kiểm duyệt."}
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Không tải được URL trong môi trường hiện tại: {exc.__class__.__name__}.") from exc
