from typing import List
from dataclasses import dataclass

@dataclass
class Citation:
    document_name: str
    page_number: int
    chunk_text: str
    relevance_score: float

def extract_citations(docs_with_scores: List) -> List[Citation]:
    citations = []
    for doc, score in docs_with_scores:
        citations.append(Citation(
            document_name=doc.metadata.get("source", "Unknown"),
            page_number=doc.metadata.get("page", 0),
            chunk_text=doc.page_content[:200] + "...",
            relevance_score=round(score, 4)
        ))
    return citations

def format_citations(citations: List[Citation]) -> str:
    formatted = "\n\n📚 **Sources:**\n"
    for i, c in enumerate(citations, 1):
        formatted += (
            f"\n[{i}] 📄 {c.document_name} "
            f"| Page {c.page_number} "
            f"| Relevance: {c.relevance_score}\n"
            f"    *\"{c.chunk_text}\"*\n"
        )
    return formatted