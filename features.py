import ipaddress
import re
from urllib.parse import urlparse


SUSPICIOUS_WORDS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification",
    "account", "update", "secure", "security", "password", "confirm",
    "confirmation", "bank", "paypal", "recover", "recovery", "credential",
    "wallet", "authorize", "authentication",
)

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly",
    "buff.ly", "bit.do", "cutt.ly", "shorturl.at",
}


def parse_url(url):
    """Return a normalized URL string and its parsed result.

    A missing scheme is interpreted as HTTPS. This does not mean the URL
    is safe or that a connection to it will succeed.
    """
    if not isinstance(url, str):
        raise ValueError("Please enter a URL as text.")

    url = url.strip()
    if not url:
        raise ValueError("Please enter a URL.")

    if any(ord(char) < 32 for char in url):
        raise ValueError("The URL contains control characters.")

    candidate = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "https://" + url

    try:
        parsed = urlparse(candidate)
        hostname = parsed.hostname
        # Accessing .port validates malformed/out-of-range ports.
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid URL: {exc}") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    if not hostname:
        raise ValueError("The URL must contain a valid hostname.")

    # Reject whitespace in hostname. urlparse itself is intentionally permissive.
    if any(char.isspace() for char in hostname):
        raise ValueError("The hostname contains whitespace.")

    return candidate, parsed, hostname.lower(), port


def _is_ip_address(hostname):
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def extract_features(url):
    """Extract the numeric features used by both training and Flask inference."""
    normalized_url, parsed_url, hostname, port = parse_url(url)
    path = parsed_url.path or ""
    query = parsed_url.query or ""
    scheme = parsed_url.scheme.lower()

    # Treat an optional leading www. as an equivalent spelling for feature
    # extraction. This makes https://example.com and https://www.example.com
    # produce the same hostname-related signals when all other URL parts match.
    # The submitted URL itself is not changed in the UI or network-checked.
    hostname_without_trailing_dot = hostname.rstrip(".")
    feature_hostname = hostname_without_trailing_dot
    if feature_hostname.startswith("www."):
        feature_hostname = feature_hostname[4:]

    # Build a feature-only canonical URL, preserving path, query, scheme and port.
    # Keep any unusual port as a feature; it is not automatically proof of phishing.
    # Preserve user-info (including @), because it can be a deceptive URL signal.
    user_info = ""
    if parsed_url.username is not None:
        user_info = parsed_url.netloc.rsplit("@", 1)[0] + "@"

    formatted_hostname = feature_hostname
    if ":" in formatted_hostname and not formatted_hostname.startswith("["):
        # IPv6 literals require brackets when represented in a URL authority.
        formatted_hostname = f"[{formatted_hostname}]"

    canonical_netloc = user_info + formatted_hostname
    if port is not None:
        canonical_netloc += f":{port}"
    feature_url = parsed_url._replace(netloc=canonical_netloc).geturl()
    url_lower = feature_url.lower()

    # Count labels before the final two hostname labels as a simple heuristic.
    # This is not a public-suffix-aware registered-domain calculation.
    hostname_without_trailing_dot = feature_hostname
    labels = hostname_without_trailing_dot.split(".") if hostname_without_trailing_dot else []

    try:
        ipaddress.ip_address(hostname_without_trailing_dot)
        has_ip = 1
    except ValueError:
        has_ip = 0

    default_port = 443 if scheme == "https" else 80
    nonstandard_port = int(port is not None and port != default_port)

    suspicious_in_url = int(any(word in url_lower for word in SUSPICIOUS_WORDS))
    suspicious_in_hostname = int(
        any(word in hostname_without_trailing_dot for word in SUSPICIOUS_WORDS)
    )

    is_shortener = int(
        any(
            hostname_without_trailing_dot == shortener
            or hostname_without_trailing_dot.endswith("." + shortener)
            for shortener in SHORTENER_DOMAINS
        )
    )

    try:
        features = {
            "url_length": len(feature_url),
            "num_dots": feature_url.count("."),
            "num_slashes": feature_url.count("/"),
            "num_hyphens": feature_url.count("-"),
            "num_at": feature_url.count("@"),
            "num_question": feature_url.count("?"),
            "num_equal": feature_url.count("="),
            "num_ampersand": feature_url.count("&"),
            "num_percent": feature_url.count("%"),
            "num_underscore": feature_url.count("_"),
            "num_semicolon": feature_url.count(";"),
            "num_colon": feature_url.count(":"),
            "num_digits": sum(char.isdigit() for char in feature_url),
            "num_letters": sum(char.isalpha() for char in feature_url),
            "https": int(scheme == "https"),
            "has_ip": has_ip,
            "domain_length": len(hostname_without_trailing_dot),
            "path_length": len(path),
            "query_length": len(query),
            "number_of_subdomains": max(len(labels) - 2, 0) if not has_ip else 0,
            "num_subdirectories": path.count("/"),
            "domain_has_hyphen": int("-" in hostname_without_trailing_dot),
            "has_suspicious_words": suspicious_in_url,
            "suspicious_word_in_domain": suspicious_in_hostname,
            "has_shortened_url": is_shortener,
            "special_character_count": len(re.findall(r"[^a-zA-Z0-9]", feature_url)),
            "digit_ratio": sum(char.isdigit() for char in feature_url) / len(feature_url),
            "letter_ratio": sum(char.isalpha() for char in feature_url) / len(feature_url),
            "has_at_symbol": int("@" in feature_url),
            "has_double_slash": int("//" in path),
            "has_nonstandard_port": nonstandard_port,
        }
    except ZeroDivisionError:
        # parse_url rejects an empty URL, but keep this guard defensive.
        features["digit_ratio"] = 0.0
        features["letter_ratio"] = 0.0

    return features
