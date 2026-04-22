"""
Fallback response module for out-of-scope queries.

When a user's question falls outside the scope of uploaded documents
(similarity score below threshold), this module generates helpful
alternative search links instead of hallucinating an answer.
"""

import re
from urllib.parse import quote, quote_plus


def fallback_response(query: str) -> dict:
    """
    Generate a structured fallback response with external search links.

    Args:
        query: The user's original question.

    Returns:
        dict with type, message, and structured links for the UI.
    """
    cleaned_query = _clean_query(query)
    encoded_query = quote_plus(cleaned_query)
    wiki_query = re.sub(r'\s+', '_', cleaned_query.strip())

    links = {
        "Perplexity AI": f"https://www.perplexity.ai/search?q={encoded_query}",
        "Google": f"https://www.google.com/search?q={encoded_query}",
        "Wikipedia": f"https://en.wikipedia.org/wiki/{quote(wiki_query)}",
    }

    message = (
        "[Out-of-Scope] This question is outside the scope of the uploaded documents.\n\n"
        "You can explore it here:\n\n"
    )
    for name, url in links.items():
        message += f"- **{name}**: [{url}]({url})\n"

    return {
        "type": "fallback",
        "message": message,
        "links": links,
        "query": cleaned_query,
    }


def _clean_query(text: str) -> str:
    """Remove special characters, keeping only alphanumeric and spaces."""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
