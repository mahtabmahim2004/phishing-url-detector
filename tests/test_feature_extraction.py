import socket

import pandas as pd

from feature_extraction import FEATURE_NAMES, extract_features, extract_features_batch


def test_normal_url_has_structural_features():
    features = extract_features("http://www.example.com/path")

    assert features["has_http"] == 1
    assert features["hostname_length"] == len("www.example.com")
    assert features["number_of_subdomains"] == 1


def test_https_url_is_detected():
    features = extract_features("https://example.com")

    assert features["has_https"] == 1
    assert features["has_http"] == 0


def test_ip_address_url_is_detected():
    features = extract_features("http://192.168.1.10/login")

    assert features["has_ip_address"] == 1


def test_at_symbol_and_suspicious_keyword_are_counted():
    features = extract_features("http://user@example.com/verify-account")

    assert features["number_of_at_symbols"] == 1
    assert features["has_suspicious_keyword"] == 1


def test_query_parameters_are_counted():
    features = extract_features("https://example.com/search?q=test&id=2#results")

    assert features["query_length"] == len("q=test&id=2")
    assert features["number_of_question_marks"] == 1
    assert features["number_of_equals"] == 2


def test_empty_and_malformed_urls_are_safe():
    empty = extract_features("")
    malformed = extract_features("http://[malformed")

    assert set(empty) == set(FEATURE_NAMES)
    assert all(value == 0 for value in empty.values())
    assert set(malformed) == set(FEATURE_NAMES)


def test_feature_keys_are_deterministic():
    assert tuple(extract_features("example.com")) == FEATURE_NAMES
    assert tuple(extract_features("https://example.com")) == FEATURE_NAMES


def test_batch_extraction_returns_ordered_dataframe():
    urls = ["https://example.com", "http://192.168.0.1/login"]

    features = extract_features_batch(urls)

    assert isinstance(features, pd.DataFrame)
    assert tuple(features.columns) == FEATURE_NAMES
    assert features["has_ip_address"].tolist() == [0, 1]


def test_extraction_does_not_open_network_connections(monkeypatch):
    def fail_if_socket_is_used(*args, **kwargs):
        raise AssertionError("network access was attempted")

    monkeypatch.setattr(socket, "socket", fail_if_socket_is_used)

    features = extract_features("https://example.com/login")

    assert features["has_https"] == 1