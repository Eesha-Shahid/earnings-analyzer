import re
from config import Config

SECTION_PATTERNS = {
    "opening_remarks": [
        r"(good\s+(morning|afternoon|evening))",
        r"(prepared\s+remarks)",
        r"(thank\s+you.*operator)",
    ],
    "financial_results": [
        r"(financial\s+results)",
        r"(revenue\s+(was|were|of))",
        r"(gross\s+margin)",
        r"(earnings\s+per\s+share)",
        r"(net\s+income)",
    ],
    "guidance": [
        r"(outlook)",
        r"(guidance)",
        r"(next\s+quarter)",
        r"(fiscal\s+year\s+\d{4})",
        r"(we\s+expect)",
    ],
    "risk_factors": [
        r"(risk)",
        r"(headwind)",
        r"(challenge)",
        r"(uncertain)",
        r"(concern)",
    ],
    "qa_session": [
        r"(question.and.answer)",
        r"(q\s*&\s*a)",
        r"(open.*questions)",
        r"(analyst.*question)",
    ],
}


def detect_section(text: str) -> str:
    text_lower = text.lower()
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return section
    return "general"


def chunk_text(
    text: str,
    chunk_size: int = Config.CHUNK_SIZE,
    overlap: int = Config.CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_document(
    text: str,
    metadata: dict,
    doc_id: str,
) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    chunk_index = 0

    for para in paragraphs:
        section = detect_section(para)

        if len(para.split()) > Config.CHUNK_SIZE:
            sub_chunks = chunk_text(para)
        else:
            sub_chunks = [para]

        for sub_chunk in sub_chunks:
            if len(sub_chunk.strip()) < 50:
                continue

            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            chunks.append({
                "id": chunk_id,
                "text": sub_chunk,
                "metadata": {
                    **metadata,
                    "section": section,
                    "chunk_index": chunk_index,
                    "word_count": len(sub_chunk.split()),
                    "contains_numbers": bool(
                        re.search(r'\$[\d,.]+|\d+%|\d+\.\d+[BMK]?', sub_chunk)
                    ),
                },
            })
            chunk_index += 1

    return chunks
