"""
Modul Web Scraper untuk Ekstraksi Konten Berita dari URL/Link
Mendukung portal berita nasional (Kompas, Detik, Tempo, TurnBackHoax, dll.)
Menggunakan requests dan BeautifulSoup4.
"""

import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup


def extract_article_from_url(url: str, timeout: int = 10) -> dict:
    """
    Mengambil dan mengekstrak judul serta isi artikel berita dari tautan URL.
    Returns dict:
      - success (bool)
      - title (str)
      - content (str)
      - full_text (str)
      - domain (str)
      - error (str or None)
    """
    if not url or not isinstance(url, str):
        return {"success": False, "error": "URL tidak boleh kosong."}

    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url

    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id,en-US;q=0.9,en;q=0.8"
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Deteksi encoding
        if response.encoding is None or response.encoding == "ISO-8859-1":
            response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, "html.parser")

        # Hapus elemen sampah yang bukan isi berita
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "svg"]):
            tag.decompose()

        # 1. Ekstraksi Judul Berita
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)

        # Bersihkan akhiran judul portal (misal: " - Kompas.com", " | Detik News")
        title = re.sub(r'\s*[-|–]\s*(Kompas\.com|detikNews|detikcom|Tempo\.co|Liputan6|TurnBackHoax).*$', '', title, flags=re.IGNORECASE).strip()

        # 2. Ekstraksi Isi Paragraf Berita
        article_body = None
        # Cek selector artikel umum
        selectors = [
            "article",
            ".detail__body-text",       # Detik
            ".read__content",           # Kompas
            ".detail-konten",           # Tempo
            ".entry-content",           # WordPress / TurnBackHoax
            ".post-content",
            ".article-content",
            "#article-body"
        ]

        for sel in selectors:
            element = soup.select_one(sel)
            if element:
                article_body = element
                break

        if article_body:
            paragraphs = [p.get_text(strip=True) for p in article_body.find_all("p") if len(p.get_text(strip=True)) > 25]
        else:
            # Fallback jika tidak menemukan wrapper artikel khusus
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 30]

        # Filter paragraf disclaimer atau iklan
        cleaned_paragraphs = []
        for p in paragraphs:
            # Hindari kalimat copyright / pagination
            if re.search(r'(baca juga|simak video|copyright|halaman selanjutnya|advertisement)', p, re.IGNORECASE):
                continue
            cleaned_paragraphs.append(p)

        content = " ".join(cleaned_paragraphs).strip()

        if not content and not title:
            return {
                "success": False,
                "error": "Gagal mengekstrak teks berita dari URL tersebut. Halaman mungkin memerlukan autentikasi atau proteksi bot."
            }

        full_text = f"{title}. {content}".strip() if content else title

        return {
            "success": True,
            "title": title,
            "content": content,
            "full_text": full_text,
            "domain": domain,
            "paragraph_count": len(cleaned_paragraphs),
            "error": None
        }

    except requests.exceptions.Timeout:
        return {"success": False, "error": f"Batas waktu koneksi habis saat mengakses {domain}."}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Gagal mengakses URL: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Terjadi kesalahan pemrosesan: {str(e)}"}


if __name__ == "__main__":
    # Test lokal scraper
    test_url = "https://turnbackhoax.id"
    res = extract_article_from_url(test_url)
    print("Test Scraper Result:", res["success"], res.get("domain"))
