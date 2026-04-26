import requests
import time


HEADERS = {
    "User-Agent": "Eesha eesha.shahid14@gmail.com",  # required by SEC
    "Accept-Encoding": "gzip, deflate",
}


class EDGARFetcher:
    BASE_URL = "https://data.sec.gov"

    def get_cik(self, ticker: str) -> str:
        """Convert ticker to SEC CIK number"""
        url = "https://www.sec.gov/files/company_tickers.json"
        res = requests.get(url, headers=HEADERS)
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
        """"8-K filings aren't all earnings calls — companies file 8-Ks for 
        press releases, executive changes, and other events too. You need to 
        filter only earnings-related 8-Ks"""
        cik = self.get_cik(ticker)
        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        res = requests.get(url, headers=HEADERS)
        data = res.json()

        filings = []
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocument", [])
        items = recent.get("items", [])  # ← add this

        for i, form in enumerate(forms):
            if form == form_type and len(filings) < limit:
                # 8-K item 2.02 = Results of Operations (earnings calls)
                # Filter out non-earnings 8-Ks
                item = items[i] if i < len(items) else ""
                if "2.02" not in str(item):
                    continue

                filings.append({
                    "ticker": ticker,
                    "cik": cik,
                    "form_type": form,
                    "filing_date": dates[i],
                    "accession_number": accessions[i].replace("-", ""),
                    "primary_doc": descriptions[i],
                })

        return filings

    def download_filing(self, filing: dict) -> str:
        """Download filing text, return content"""
        cik = filing["cik"]
        accession = filing["accession_number"]
        doc = filing["primary_doc"]

        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession}/{doc}"
        )

        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            return ""

        time.sleep(0.1)
        return res.text

    def fetch_transcript(self, ticker: str, limit: int = 8) -> list[dict]:
        """Fetch last N earnings transcripts for a ticker"""
        filings = self.get_filings(ticker, form_type="8-K", limit=limit)
        transcripts = []

        for filing in filings:
            content = self.download_filing(filing)
            if content:
                transcripts.append({
                    **filing,
                    "content": content,
                    "source_url": (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{int(filing['cik'])}/{filing['accession_number']}/"
                        f"{filing['primary_doc']}"
                    ),
                    "source": "sec_edgar",
                })
            time.sleep(0.2)

        return transcripts
