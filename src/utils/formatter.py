# src/utils/formatter.py

def format_response(raw_answer: str, snippets: list[dict]) -> str:
    md = raw_answer.strip()
    seen = set()
    sources = []
    for s in snippets:
        src = s["source"]
        if src not in seen:
            seen.add(src)
            sources.append(src)

    if sources:
        md += "\n\n---\n**References:**\n"
        for url in sources:
            md += f"- {url}\n"
    return md
