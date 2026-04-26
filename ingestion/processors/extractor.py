import fitz  # PyMuPDF
import pdfplumber
import io
import re
from PIL import Image


class DocumentExtractor:

    def extract(self, content: str | bytes, source_url: str = "") -> dict:
        """
        Main method. Handles both HTML (EDGAR) and PDF.
        Returns structured content with text_blocks, tables, images.
        """
        if isinstance(content, bytes) and content[:4] == b"%PDF":
            return self._extract_pdf(content)
        else:
            return self._extract_html(content if isinstance(content, str) else content.decode("utf-8", errors="ignore"))

    # ─── HTML Extraction ────────────────────────────────────────────────

    def _extract_html(self, html: str) -> dict:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        result = {
            "text_blocks": [],
            "tables": [],
            "images": [],
        }

        for table in soup.find_all("table"):
            table_data = self._parse_html_table(table)
            if table_data and self._is_financial_table(table_data):
                result["tables"].append({
                    "data": table_data,
                    "text": self._table_to_text(table_data),
                    "page": None,
                })
            table.replace_with(soup.new_string(self._table_to_text(table_data)))

        for tag in soup.find_all(["p", "div", "span", "li"]):
            text = tag.get_text(separator=" ", strip=True)
            if len(text.split()) > 10:
                result["text_blocks"].append(text)

        return result

    def _parse_html_table(self, table_tag) -> list[list[str]]:
        rows = []
        for tr in table_tag.find_all("tr"):
            cells = [
                td.get_text(separator=" ", strip=True)
                for td in tr.find_all(["td", "th"])
            ]
            if any(cells):
                rows.append(cells)
        return rows

    def _is_financial_table(self, table: list[list[str]]) -> bool:
        if not table:
            return False
        flat = " ".join([" ".join(row) for row in table]).lower()
        financial_keywords = [
            "revenue", "income", "margin", "earnings", "eps",
            "cash", "debt", "assets", "liabilities", "quarter",
            "fiscal", "billion", "million", "%", "$",
        ]
        matches = sum(1 for kw in financial_keywords if kw in flat)
        return matches >= 2

    def _table_to_text(self, table: list[list[str]]) -> str:
        if not table:
            return ""

        lines = ["[TABLE START]"]
        if len(table) > 0:
            headers = " | ".join(table[0])
            lines.append(f"Headers: {headers}")

        for row in table[1:]:
            if any(row):
                lines.append(" | ".join(cell for cell in row if cell))

        lines.append("[TABLE END]")
        return "\n".join(lines)

    # ─── PDF Extraction ─────────────────────────────────────────────────

    def _extract_pdf(self, content: bytes) -> dict:
        result = {
            "text_blocks": [],
            "tables": [],
            "images": [],
        }

        # pdfplumber for tables
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for table in page.extract_tables():
                    if table and self._is_financial_table(table):
                        result["tables"].append({
                            "data": table,
                            "text": self._table_to_text(table),
                            "page": page_num + 1,
                        })

        # PyMuPDF for text + images
        doc = fitz.open(stream=content, filetype="pdf")

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                result["text_blocks"].append(text)

            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                result["images"].append({
                    "bytes": base_image["image"],
                    "ext": base_image["ext"],
                    "page": page_num + 1,
                    "index": img_index,
                })

        doc.close()
        return result
