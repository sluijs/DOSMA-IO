import re
from typing import List, Union, Optional

__all__ = ["is_url", "extract_boundary", "extract_part"]


def is_url(url: str) -> bool:
    """Check if a string represents a valid URL.

    Args:
        url (str): URL.

    Returns:
        bool: Result of the URL validation.
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return re.match(regex, url) is not None


def extract_part(part: bytes) -> Union[bytes, None]:
    """Extract part from multipart stream."""
    if part in [b"", b"--", b"\r\n"] or part.startswith(b"--\r\n"):
        return None

    idx = part.index(b"\r\n\r\n")
    if idx > -1:
        return part[idx + 4:]

    raise ValueError("Part is not CRLF CRLF terminated.")


def extract_boundary(content_info: List[str]) -> Optional[bytes]:
    """Extract boundary from content info."""
    for item in content_info:
        if '=' not in item:
            continue

        key, value = item.split('=', maxsplit=1)
        if key.lower() == "boundary":
            return b"--" + value.strip('"').encode("utf-8")

    return None
