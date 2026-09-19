"""Safe lexical feature extraction for URL strings.

All analysis is local. This module never requests, resolves, or otherwise
accesses a URL over the network.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable
from urllib.parse import urlsplit

import pandas as pd


FEATURE_NAMES = (
	"url_length",
	"hostname_length",
	"path_length",
	"query_length",
	"number_of_dots",
	"number_of_hyphens",
	"number_of_digits",
	"number_of_special_characters",
	"number_of_slashes",
	"number_of_question_marks",
	"number_of_equals",
	"number_of_at_symbols",
	"number_of_percent_symbols",
	"number_of_subdomains",
	"has_ip_address",
	"has_https",
	"has_http",
	"has_url_shortener_pattern",
	"has_suspicious_keyword",
)

_URL_SHORTENER_DOMAINS = {
	"bit.ly",
	"bitly.com",
	"buff.ly",
	"goo.gl",
	"is.gd",
	"ow.ly",
	"rebrand.ly",
	"shorturl.at",
	"t.co",
	"tinyurl.com",
}
_SUSPICIOUS_KEYWORDS = (
	"account",
	"bank",
	"confirm",
	"credential",
	"login",
	"password",
	"secure",
	"signin",
	"update",
	"verify",
)


def _parse_hostname(url: str) -> tuple[str, str]:
	"""Return a safely parsed hostname and scheme for a URL-like string."""
	candidate = url if "://" in url else f"//{url}"
	try:
		parsed = urlsplit(candidate)
		return (parsed.hostname or "").lower(), parsed.scheme.lower()
	except ValueError:
		return "", ""


def _is_ip_address(hostname: str) -> int:
	try:
		ipaddress.ip_address(hostname)
	except ValueError:
		return 0
	return 1


def _has_shortener_pattern(hostname: str) -> int:
	return int(
		hostname in _URL_SHORTENER_DOMAINS
		or any(hostname.endswith(f".{domain}") for domain in _URL_SHORTENER_DOMAINS)
	)


def extract_features(url: str) -> dict[str, int]:
	"""Extract explainable lexical and structural features from one URL string."""
	value = "" if url is None else str(url).strip()
	hostname, scheme = _parse_hostname(value)
	candidate = value if "://" in value else f"//{value}"
	try:
		parsed = urlsplit(candidate)
		path = parsed.path
		query = parsed.query
	except ValueError:
		path = ""
		query = ""

	hostname_parts = hostname.rstrip(".").split(".") if hostname else []
	subdomain_count = max(0, len(hostname_parts) - 2)
	lower_value = value.lower()

	return {
		"url_length": len(value),
		"hostname_length": len(hostname),
		"path_length": len(path),
		"query_length": len(query),
		"number_of_dots": value.count("."),
		"number_of_hyphens": value.count("-"),
		"number_of_digits": sum(character.isdigit() for character in value),
		"number_of_special_characters": sum(
			not character.isalnum() for character in value
		),
		"number_of_slashes": value.count("/"),
		"number_of_question_marks": value.count("?"),
		"number_of_equals": value.count("="),
		"number_of_at_symbols": value.count("@"),
		"number_of_percent_symbols": value.count("%"),
		"number_of_subdomains": subdomain_count,
		"has_ip_address": _is_ip_address(hostname),
		"has_https": int(scheme == "https"),
		"has_http": int(scheme == "http"),
		"has_url_shortener_pattern": _has_shortener_pattern(hostname),
		"has_suspicious_keyword": int(
			any(re.search(rf"(?<![a-z]){re.escape(keyword)}(?![a-z])", lower_value)
				for keyword in _SUSPICIOUS_KEYWORDS)
		),
	}


def extract_features_batch(urls: Iterable[str]) -> pd.DataFrame:
	"""Extract features for an iterable of URLs with stable column ordering."""
	rows = [extract_features(url) for url in urls]
	return pd.DataFrame(rows, columns=FEATURE_NAMES)
