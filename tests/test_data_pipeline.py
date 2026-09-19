import pandas as pd
import pytest

from train_model import DatasetValidationError, load_url_dataset, prepare_train_test_split


def write_dataset(tmp_path, content):
    path = tmp_path / "urls.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_csv_loading(tmp_path):
    path = write_dataset(tmp_path, "url,label\nhttps://example.com,0\nhttps://bad.test,1\n")

    records = load_url_dataset(path)

    assert list(records.columns) == ["url", "label"]
    assert records.to_dict("records") == [
        {"url": "https://example.com", "label": 0},
        {"url": "https://bad.test", "label": 1},
    ]


def test_missing_required_column(tmp_path):
    path = write_dataset(tmp_path, "url\nhttps://example.com\n")

    with pytest.raises(DatasetValidationError, match="label"):
        load_url_dataset(path)


def test_missing_url_is_removed(tmp_path):
    path = write_dataset(tmp_path, "url,label\n,0\nhttps://example.com,1\n")

    records, report = load_url_dataset(path, return_report=True)

    assert records["url"].tolist() == ["https://example.com"]
    assert report.removed_records == 1


def test_duplicate_url_is_removed(tmp_path):
    path = write_dataset(
        tmp_path,
        "url,label\nhttps://example.com,0\nhttps://example.com,0\n",
    )

    records = load_url_dataset(path)

    assert len(records) == 1


def test_invalid_label_is_rejected(tmp_path):
    path = write_dataset(tmp_path, "url,label\nhttps://example.com,2\n")

    with pytest.raises(DatasetValidationError, match="only 0.*1"):
        load_url_dataset(path)


def test_whitespace_is_cleaned(tmp_path):
    path = write_dataset(tmp_path, "url,label\n  https://example.com  , 0 \n")

    records = load_url_dataset(path)

    assert records.iloc[0].to_dict() == {"url": "https://example.com", "label": 0}


def test_train_test_split_is_deterministic():
    records = pd.DataFrame(
        {
            "url": [f"https://example-{index}.test" for index in range(10)],
            "label": [0, 1] * 5,
        }
    )

    first = prepare_train_test_split(records, test_size=0.3)
    second = prepare_train_test_split(records, test_size=0.3)

    for first_part, second_part in zip(first, second):
        assert first_part.equals(second_part)