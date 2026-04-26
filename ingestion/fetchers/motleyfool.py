import requests
import re
import time
from bs4 import BeautifulSoup
from datetime import datetime


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class MotleyFoolFetcher:
    BASE_URL = "https://www.fool.com"
    SEARCH_URL = "https://www.fool.com/search/solr.aspx"

    def search_transcripts(self, ticker: str, limit: int = 8) -> list[str]:
        """Search for earnings call transcript article URLs"""
        params = {
            "q": f"{ticker} earnings call transcript",
            "type": "13",
            "page": "1",
        }

        try:
            res = requests.get(
                self.SEARCH_URL,
                params=params,
                headers=HEADERS,
                timeout=10,
            )
            res.raise_for_status()
        except Exception as e:
            print(f"Motley Fool search failed for {ticker}: {e}")
            return []

        soup = BeautifulSoup(res.text, "html.parser")

        results = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True).lower()

            if (
                "earnings-call-transcript" in href
                or ("transcript" in text and ticker.lower() in text)
            ):
                full_url = (
                    href if href.startswith("http") else self.BASE_URL + href
                )
                results.append(full_url)

            if len(results) >= limit:
                break

        return results[:limit]

    def scrape_transcript(self, url: str) -> dict | None:
        """Scrape a single transcript page"""
        try:
            time.sleep(1)
            res = requests.get(url, headers=HEADERS, timeout=15)
            res.raise_for_status()
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            return None

        soup = BeautifulSoup(res.text, "html.parser")

        # Extract title
        title = ""
        title_tag = soup.find("h1")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Extract date
        filing_date = ""
        date_tag = soup.find("time")
        if date_tag:
            dt_str = date_tag.get("datetime", "")
            try:
                dt = datetime.fromisoformat(dt_str[:10])
                filing_date = dt.strftime("%Y-%m-%d")
            except Exception:
                filing_date = dt_str[:10]

        # Extract main article content
        content = ""
        article = soup.find("article") or soup.find(
            "div", class_=re.compile(r"article|content|body", re.I)
        )
        if article:
            for tag in article(["aside", "nav", "figure", "script"]):
                tag.decompose()
            content = article.get_text(separator="\n", strip=True)

        if not content or len(content) < 500:
            return None

        quarter, year = self._parse_quarter_from_title(title)

        return {
            "title": title,
            "content": content,
            "filing_date": filing_date,
            "quarter": quarter,
            "year": year,
            "source_url": url,
            "source": "motley_fool",
        }

    def _parse_quarter_from_title(self, title: str) -> tuple[str, str]:
        quarter_match = re.search(r"Q([1-4])", title, re.I)
        year_match = re.search(r"(20\d{2})", title)

        quarter = f"Q{quarter_match.group(1)}" if quarter_match else ""
        year = year_match.group(1) if year_match else ""

        return quarter, year

    def fetch_transcripts(self, ticker: str, limit: int = 8) -> list[dict]:
        """Main method — search and scrape transcripts for a ticker"""
        print(f"  Fetching Motley Fool transcripts for {ticker}...")

        urls = self.search_transcripts(ticker, limit=limit)
        if not urls:
            print(f"  No Motley Fool transcripts found for {ticker}")
            return []

        transcripts = []
        for url in urls:
            transcript = self.scrape_transcript(url)
            if transcript:
                transcript["ticker"] = ticker
                transcripts.append(transcript)
                print(f"  Scraped: {transcript['title'][:60]}...")

        return transcripts
