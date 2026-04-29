import os
import hashlib
import requests
import time
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Eesha eesha.shahid14@gmail.com",  # required by SEC
    "Accept-Encoding": "gzip, deflate",
}

CACHE_DIR = "./data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class EDGARFetcher:
    BASE_URL = "https://data.sec.gov"

    def get_cik(self, ticker: str) -> str:
        url = "https://www.sec.gov/files/company_tickers.json"
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
        raise ValueError(f"CIK not found for {ticker}")

    def get_filings(
        self,
        ticker: str,
        form_type: str = "8-K",
        limit: int = 8,
    ) -> list[dict]:
        cik = self.get_cik(ticker)
        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()

        filings = []
        recent     = data.get("filings", {}).get("recent", {})
        forms      = recent.get("form", [])
        dates      = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        items      = recent.get("items", [])

        for i, form in enumerate(forms):
            if form != form_type:
                continue
            if len(filings) >= limit:
                break

            # Only Item 2.02 = Results of Operations (actual earnings releases)
            item = str(items[i]) if i < len(items) else ""
            if "2.02" not in item:
                continue

            filings.append({
                "ticker":           ticker,
                "cik":              cik,
                "form_type":        form,
                "filing_date":      dates[i],
                "accession_number": accessions[i].replace("-", ""),
            })

        return filings

    def _get_filing_index(self, cik: str, accession: str) -> list[dict]:
        """
        Fetch the filing index page and return all documents listed.
        Returns list of {type, description, url}
        """
        acc_formatted = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession}/{acc_formatted}-index.htm"
        )

        try:
            res = requests.get(index_url, headers=HEADERS, timeout=10)
            if res.status_code != 200:
                print(f"    Index fetch failed: {res.status_code}")
                return []
        except Exception as e:
            print(f"    Index fetch error: {e}")
            return []

        soup = BeautifulSoup(res.text, "html.parser")
        docs = []

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            link_tag = row.find("a", href=True)
            if not link_tag:
                continue

            href = link_tag["href"]
            full_url = (
                "https://www.sec.gov" + href
                if href.startswith("/")
                else href
            )

            doc_type    = cells[3].get_text(strip=True) if len(cells) > 3 else ""
            description = cells[1].get_text(strip=True) if len(cells) > 1 else ""

            docs.append({
                "type":        doc_type,
                "description": description,
                "url":         full_url,
            })

        return docs

    def _fetch_exhibit(self, cik: str, accession: str) -> str:
        """
        Find and download Exhibit 99.1 (earnings press release) from the filing.
        Falls back progressively if not found.
        """
        docs = self._get_filing_index(cik, accession)

        if not docs:
            print(f"    Could not get filing index for {accession}")
            return ""

        # Priority 1: EX-99.1 by type field
        for doc in docs:
            doc_type = doc["type"].upper()
            if "EX-99.1" in doc_type or doc_type == "99.1":
                print(f"    Found EX-99.1 by type: {doc['url']}")
                return self._download_url(doc["url"])

        # Priority 2: Match by description
        for doc in docs:
            desc = doc["description"].lower()
            if "99.1" in desc or "press release" in desc or "earnings" in desc:
                print(f"    Found by description: {doc['url']}")
                return self._download_url(doc["url"])

        # Priority 3: Any HTM that isn't the index
        for doc in docs:
            url_lower = doc["url"].lower()
            if (url_lower.endswith(".htm") or url_lower.endswith(".html")) \
                    and "index" not in url_lower:
                print(f"    Fallback to HTM: {doc['url']}")
                return self._download_url(doc["url"])

        print(f"    No suitable exhibit found in {len(docs)} documents")
        return ""

    def _download_url(self, url: str) -> str:
        # Check cache first
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = f"{CACHE_DIR}/{cache_key}.html"

        if os.path.exists(cache_path):
            print(f"    Cache hit: {url[-50:]}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
            
        # Download and cache
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            res.raise_for_status()
            time.sleep(0.15)
            content = res.text

            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)

            return content
        except Exception as e:
            print(f"    Download failed for {url}: {e}")
            return ""

    def fetch_transcript(self, ticker: str, limit: int = 8) -> list[dict]:
        """Main method — fetch last N earnings press releases for a ticker"""
        print(f"  Fetching EDGAR filings for {ticker}...")
        filings = self.get_filings(ticker, form_type="8-K", limit=limit)
        print(f"  Found {len(filings)} Item 2.02 filings")

        transcripts = []
        for filing in filings:
            cik       = filing["cik"]
            accession = filing["accession_number"]

            print(f"  Processing {ticker} {filing['filing_date']} ({accession[:12]}...)...")
            content = self._fetch_exhibit(cik, accession)

            if not content or len(content) < 1000:
                print(f"    Skipping — insufficient content ({len(content)} chars)")
                continue

            transcripts.append({
                **filing,
                "content":    content,
                "source_url": (
                    f"https://www.sec.gov/cgi-bin/browse-edgar?"
                    f"action=getcompany&CIK={cik}&type=8-K"
                ),
                "source": "sec_edgar",
            })
            print(f"    ✅ Got {len(content):,} chars of content")
            time.sleep(0.3)

        return transcripts