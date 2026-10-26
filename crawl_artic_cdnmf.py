"""Download and prepare Art Institute of Chicago artworks for CDNMF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import re
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


API_ROOT = "https://api.artic.edu/api/v1"
FIELDS = (
    "id",
    "artist_id",
    "title",
    "artist_title",
    "date_display",
    "place_of_origin",
    "medium_display",
    "department_title",
    "classification_title",
    "category_titles",
    "artwork_type_title",
    "style_title",
    "subject_titles",
    "technique_titles",
    "material_titles",
    "term_titles",
)
TAXONOMY_FIELDS = (
    "classification_titles",
    "category_titles",
    "style_titles",
    "subject_titles",
    "technique_titles",
    "material_titles",
)
CSV_FIELDS = tuple(dict.fromkeys((*FIELDS, *TAXONOMY_FIELDS)))
BASE_FEATURE_FIELDS = (
    "material_titles",
    "technique_titles",
    "place_of_origin",
    "medium_display",
)
OPTIONAL_FEATURE_FIELDS = ("subject_titles", "style_titles")
LOGGER = logging.getLogger("crawl_artic_cdnmf")


def _field_value(record: dict[str, Any], field: str) -> Any:
    aliases = {
        "classification_titles": ("classification_title",),
        "style_titles": ("style_title",),
    }
    if field in record:
        return record[field]
    for alias in aliases.get(field, ()):
        if alias in record:
            return record[alias]
    return None


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(_as_text(value))).casefold()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _csv_row(record: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in CSV_FIELDS:
        value = _field_value(record, field)
        if isinstance(value, (list, dict)):
            row[field] = json.dumps(value, ensure_ascii=False)
        else:
            row[field] = _as_text(value)
    return row


def _has_required_artwork_data(record: dict[str, Any]) -> bool:
    artwork_id = _as_text(record.get("id"))
    if not artwork_id.isdigit():
        return False
    return all(
        _as_text(record.get(field))
        for field in ("artist_id", "title", "artwork_type_title")
    )


def _api_page(page: int, limit: int, retries: int = 5) -> dict[str, Any]:
    params = urlencode(
        {
            "fields": ",".join(FIELDS),
            "limit": limit,
            "page": page,
        }
    )
    request = Request(
        f"{API_ROOT}/artworks?{params}",
        headers={"User-Agent": "CDNMF-artwork-dataset/1.0"},
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=45) as response:
                payload = json.load(response)
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise ValueError("Art Institute API returned an unexpected response.")
            return payload
        except HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise RuntimeError(
                    f"Art Institute API returned HTTP {error.code} on page {page}."
                ) from error
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Art Institute API failed after {retries} attempts on page {page}."
                ) from error
            time.sleep(min(2**attempt, 30))
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Could not download or decode API page {page} after {retries} attempts."
                ) from error
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Could not retrieve API page {page}.")


def _read_crawl_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.is_file():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read crawl checkpoint: {checkpoint_path}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid crawl checkpoint format: {checkpoint_path}")
    return checkpoint


def _write_crawl_checkpoint(checkpoint_path: Path, state: dict[str, Any]) -> None:
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)


def _existing_artwork_ids(raw_path: Path) -> set[str]:
    if not raw_path.is_file():
        return set()
    with raw_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError(
                f"Existing crawl CSV has an unexpected header: {raw_path}. "
                "Move it aside before starting a new crawl."
            )
        return {
            _as_text(row.get("id"))
            for row in reader
            if _as_text(row.get("id"))
        }


def crawl_artworks(limit: int, raw_path: Path) -> int:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = raw_path.with_suffix(".checkpoint.json")
    checkpoint = _read_crawl_checkpoint(checkpoint_path)
    seen_ids = _existing_artwork_ids(raw_path)
    retained = len(seen_ids)
    scanned = int(checkpoint.get("scanned", 0)) if checkpoint else 0
    page = int(checkpoint.get("next_page", 1)) if checkpoint else 1

    if checkpoint:
        previous_target = int(checkpoint.get("target", limit))
        if previous_target != limit and not checkpoint.get("completed", False):
            raise ValueError(
                f"Checkpoint target is {previous_target:,}, but requested {limit:,}. "
                "Continue with the original target or move the CSV/checkpoint aside."
            )
        if checkpoint.get("completed", False) and retained >= limit:
            LOGGER.info(
                "Crawl checkpoint already complete: %s artworks in %s",
                retained,
                raw_path,
            )
            return retained
    elif raw_path.is_file() and retained:
        raise ValueError(
            f"Found {retained} existing rows without a checkpoint. "
            "Move the CSV aside or restore its checkpoint before crawling."
        )

    file_exists = raw_path.is_file() and raw_path.stat().st_size > 0
    mode = "a" if file_exists else "w"
    with raw_path.open(mode, encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
            output.flush()
        _write_crawl_checkpoint(
            checkpoint_path,
            {
                "raw_csv": str(raw_path),
                "target": limit,
                "next_page": page,
                "retained": retained,
                "scanned": scanned,
                "completed": False,
            },
        )
        while retained < limit:
            page_limit = 100
            payload = _api_page(page, page_limit)
            records = payload["data"]
            if not records:
                output.flush()
                _write_crawl_checkpoint(
                    checkpoint_path,
                    {
                        "raw_csv": str(raw_path),
                        "target": limit,
                        "next_page": page,
                        "retained": retained,
                        "scanned": scanned,
                        "completed": True,
                    },
                )
                break
            for record in records:
                scanned += 1
                if not _has_required_artwork_data(record):
                    continue
                artwork_id = _as_text(record["id"])
                if artwork_id in seen_ids:
                    continue
                seen_ids.add(artwork_id)
                writer.writerow(_csv_row(record))
                retained += 1
                if retained >= limit:
                    break
            pagination = payload.get("pagination") or {}
            total_pages = pagination.get("total_pages")
            LOGGER.info(
                "Scanned %s artworks; retained %s/%s with all required fields",
                scanned,
                retained,
                limit,
            )
            if len(records) < page_limit or (total_pages and page >= total_pages):
                output.flush()
                _write_crawl_checkpoint(
                    checkpoint_path,
                    {
                        "raw_csv": str(raw_path),
                        "target": limit,
                        "next_page": page + 1,
                        "retained": retained,
                        "scanned": scanned,
                        "completed": True,
                    },
                )
                break
            output.flush()
            _write_crawl_checkpoint(
                checkpoint_path,
                {
                    "raw_csv": str(raw_path),
                    "target": limit,
                    "next_page": page if retained >= limit else page + 1,
                    "retained": retained,
                    "scanned": scanned,
                    "completed": retained >= limit,
                },
            )
            if retained >= limit:
                break
            page += 1
            time.sleep(0.15)
    if retained >= limit:
        checkpoint_state = _read_crawl_checkpoint(checkpoint_path) or {}
        checkpoint_state["completed"] = True
        checkpoint_state["retained"] = retained
        checkpoint_state["scanned"] = scanned
        checkpoint_state["next_page"] = page
        _write_crawl_checkpoint(checkpoint_path, checkpoint_state)
    LOGGER.info(
        "Crawl finished: scanned %s; retained %s complete artworks; checkpoint: %s",
        scanned,
        retained,
        checkpoint_path,
    )
    return retained


def read_artwork_csv(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    json_fields = set(TAXONOMY_FIELDS) | {"term_titles"}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"id", "title", "artwork_type_title"}.issubset(reader.fieldnames):
            raise ValueError(f"CSV is missing required artwork columns: {path}")
        for row_number, row in enumerate(reader, start=1):
            if limit is not None and row_number > limit:
                break
            record: dict[str, Any] = dict(row)
            for field in json_fields:
                value = record.get(field, "")
                if value:
                    try:
                        record[field] = json.loads(value)
                    except json.JSONDecodeError:
                        record[field] = [value]
            for normalized, original in (
                ("classification_titles", "classification_title"),
                ("style_titles", "style_title"),
            ):
                if not record.get(normalized) and record.get(original):
                    record[normalized] = record[original]
            yield record


def _stable_record_key(record: dict[str, Any]) -> bytes:
    return hashlib.sha256(_as_text(record.get("id")).encode("utf-8")).digest()


def _prune_small_artists(
    records: list[dict[str, Any]], min_per_artist: int
) -> list[dict[str, Any]]:
    artist_counts: dict[str, int] = defaultdict(int)
    for record in records:
        artist_counts[_as_text(record["artist_id"])] += 1
    retained_artists = {
        artist_id
        for artist_id, count in artist_counts.items()
        if count >= min_per_artist
    }
    return [
        record
        for record in records
        if _as_text(record["artist_id"]) in retained_artists
    ]


def _balance_records(
    records: list[dict[str, Any]],
    label_field: str,
    min_classes: int,
    max_classes: int,
    min_class_samples: int,
    max_class_ratio: float,
    max_per_artist: int,
    min_per_artist: int,
) -> list[dict[str, Any]]:
    by_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_artist[_as_text(record["artist_id"])].append(record)

    capped: list[dict[str, Any]] = []
    for artist_records in by_artist.values():
        capped.extend(
            sorted(artist_records, key=_stable_record_key)[:max_per_artist]
        )
    capped = _prune_small_artists(capped, min_per_artist)

    label_counts: dict[str, int] = defaultdict(int)
    for record in capped:
        label_counts[_as_text(record[label_field])] += 1
    selected_labels = {
        label
        for label, _ in sorted(
            (
                (label, count)
                for label, count in label_counts.items()
                if count >= min_class_samples
            ),
            key=lambda item: (-item[1], item[0].casefold()),
        )[:max_classes]
    }
    if len(selected_labels) < min_classes:
        raise ValueError(
            f"Only {len(selected_labels)} artwork types have at least "
            f"{min_class_samples} artworks; at least {min_classes} are required."
        )
    balanced = [
        record
        for record in capped
        if _as_text(record[label_field]) in selected_labels
    ]
    balanced = _prune_small_artists(balanced, min_per_artist)

    while True:
        label_counts = defaultdict(int)
        for record in balanced:
            label_counts[_as_text(record[label_field])] += 1
        small_labels = {
            label
            for label, count in label_counts.items()
            if count < min_class_samples
        }
        if small_labels:
            balanced = [
                record
                for record in balanced
                if _as_text(record[label_field]) not in small_labels
            ]
            balanced = _prune_small_artists(balanced, min_per_artist)
            continue
        if len(label_counts) < min_classes:
            raise ValueError(
                "Balancing left fewer than "
                f"{min_classes} artwork types with at least "
                f"{min_class_samples} artworks each."
            )

        smallest_count = min(label_counts.values())
        largest_label, largest_count = min(
            label_counts.items(), key=lambda item: (-item[1], item[0].casefold())
        )
        target_count = int(max_class_ratio * smallest_count)
        if largest_count <= target_count:
            return balanced

        artist_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in balanced:
            artist_records[_as_text(record["artist_id"])].append(record)
        candidates = sorted(
            (
                (artist_id, artist_rows)
                for artist_id, artist_rows in artist_records.items()
                if any(
                    _as_text(row[label_field]) == largest_label
                    for row in artist_rows
                )
            ),
            key=lambda item: (
                -sum(_as_text(row[label_field]) == largest_label for row in item[1]),
                item[0],
            ),
        )

        removed_artist: str | None = None
        for artist_id, rows in candidates:
            contributions: dict[str, int] = defaultdict(int)
            for row in rows:
                contributions[_as_text(row[label_field])] += 1
            if largest_count - contributions[largest_label] < min_class_samples:
                continue
            if any(
                count - contributions.get(label, 0) < min_class_samples
                for label, count in label_counts.items()
                if label != largest_label
            ):
                continue
            balanced = [
                record
                for record in balanced
                if _as_text(record["artist_id"]) != artist_id
            ]
            removed_artist = artist_id
            break
        if removed_artist is not None:
            balanced = _prune_small_artists(balanced, min_per_artist)
            continue

        removable_artist: str | None = None
        removable_count = 0
        for artist_id, rows in candidates:
            own_label_rows = [
                row for row in rows if _as_text(row[label_field]) == largest_label
            ]
            non_target_count = len(rows) - len(own_label_rows)
            can_remove = min(len(own_label_rows), len(rows) - min_per_artist)
            remaining_count = len(rows) - can_remove
            if can_remove > removable_count and remaining_count >= min_per_artist:
                removable_artist = artist_id
                removable_count = can_remove
        required_reduction = largest_count - target_count
        if removable_artist is None or removable_count == 0:
            raise ValueError(
                "Could not meet the maximum class ratio without violating the "
                f"{min_per_artist}-artworks-per-artist or {min_class_samples}-"
                "artworks-per-class minimum."
            )

        rows = artist_records[removable_artist]
        target_rows = sorted(
            (row for row in rows if _as_text(row[label_field]) == largest_label),
            key=_stable_record_key,
        )
        remove_count = min(required_reduction, removable_count)
        remove_ids = {_as_text(row["id"]) for row in target_rows[:remove_count]}
        balanced = [
            record for record in balanced if _as_text(record["id"]) not in remove_ids
        ]
        balanced = _prune_small_artists(balanced, min_per_artist)


def prepare_dataset(
    records: Iterable[dict[str, Any]],
    output_dir: Path,
    processed_limit: int,
    label_field: str,
    feature_limit: int,
    neighbors: int,
    min_classes: int = 6,
    max_classes: int = 8,
    min_class_samples: int = 100,
    max_class_ratio: float = 3.0,
    max_per_artist: int = 200,
    min_per_artist: int = 11,
    optional_feature_coverage: float = 0.5,
) -> dict[str, int]:
    if processed_limit < 2 or feature_limit < 1 or neighbors < 2:
        raise ValueError(
            "processed_limit must be at least 2, feature_limit positive, "
            "and neighbors at least 2."
        )
    if min_classes < 2 or max_classes < min_classes:
        raise ValueError(
            "max_classes must be at least min_classes, which must be 2 or more."
        )
    if min_class_samples < 1 or max_class_ratio <= 1:
        raise ValueError(
            "min_class_samples must be positive and max_class_ratio greater than 1."
        )
    if min_per_artist < 1 or max_per_artist < min_per_artist:
        raise ValueError("max_per_artist must be at least min_per_artist.")
    if not 0 <= optional_feature_coverage <= 1:
        raise ValueError("optional_feature_coverage must be between 0 and 1.")

    usable: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    dropped = 0
    for record in records:
        artwork_id = _as_text(record.get("id"))
        label = _as_text(record.get(label_field))
        if not _has_required_artwork_data(record) or not label:
            dropped += 1
            continue
        if artwork_id in seen_ids:
            dropped += 1
            continue
        seen_ids.add(artwork_id)
        usable.append(record)

    if len(usable) > processed_limit:
        usable = sorted(usable, key=_stable_record_key)[:processed_limit]
        dropped += len(seen_ids) - len(usable)
    selected = _balance_records(
        usable,
        label_field=label_field,
        min_classes=min_classes,
        max_classes=max_classes,
        min_class_samples=min_class_samples,
        max_class_ratio=max_class_ratio,
        max_per_artist=max_per_artist,
        min_per_artist=min_per_artist,
    )
    dropped += len(usable) - len(selected)

    if len(selected) < 2:
        raise ValueError(
            f"Only {len(selected)} usable artworks found; at least 2 are required."
        )

    labels = sorted({_as_text(record[label_field]) for record in selected})
    label_ids = {label: index for index, label in enumerate(labels)}
    feature_coverage = {
        field: sum(bool(_as_text(_field_value(record, field))) for record in selected)
        / len(selected)
        for field in OPTIONAL_FEATURE_FIELDS
    }
    optional_field = max(
        OPTIONAL_FEATURE_FIELDS,
        key=lambda field: (feature_coverage[field], field == "subject_titles"),
    )
    feature_fields = BASE_FEATURE_FIELDS + (
        (optional_field,)
        if feature_coverage[optional_field] >= optional_feature_coverage
        else ()
    )
    texts = [
        normalize_text(
            " ".join(_as_text(_field_value(record, field)) for field in feature_fields)
        )
        for record in selected
    ]
    vectorizer = TfidfVectorizer(
        max_features=feature_limit,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )
    try:
        features = vectorizer.fit_transform(texts).tocsr()
    except ValueError as error:
        raise ValueError(
            "No usable text vocabulary was found; check the downloaded API fields."
        ) from error
    if features.shape[1] == 0:
        raise ValueError("Preprocessing produced no feature columns.")

    node_ids = [_as_text(record["id"]) for record in selected]
    edge_weights: dict[tuple[int, int], float] = {}
    artist_nodes: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(selected):
        artist_nodes[_as_text(record["artist_id"])].append(index)
    for indices in artist_nodes.values():
        if len(indices) <= neighbors + 1:
            offsets = range(1, len(indices))
        else:
            offsets = range(1, neighbors // 2 + 1)
        for offset in offsets:
            for position, source in enumerate(indices):
                target = indices[(position + offset) % len(indices)]
                edge = (min(source, target), max(source, target))
                edge_weights[edge] = 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.txt"
    with feature_path.open("w", encoding="utf-8", newline="\n") as output:
        for index, artwork_id in enumerate(node_ids):
            values = features.getrow(index).toarray().ravel()
            output.write(
                artwork_id
                + " "
                + " ".join(format(float(value), ".7g") for value in values)
                + "\n"
            )

    edge_path = output_dir / "edges.txt"
    with edge_path.open("w", encoding="utf-8", newline="\n") as output:
        for (source, target), weight in sorted(edge_weights.items()):
            output.write(
                f"{node_ids[source]} {node_ids[target]} {weight:.7g}\n"
            )

    node_rows = [
        {
            "node_index": index,
            "artwork_id": artwork_id,
            "artist_id": _as_text(record.get("artist_id")),
            "title": _as_text(record.get("title")),
            "artist_title": _as_text(record.get("artist_title")),
            "label_id": label_ids[_as_text(record[label_field])],
            "artwork_type_title": _as_text(record.get("artwork_type_title")),
        }
        for index, (record, artwork_id) in enumerate(zip(selected, node_ids))
    ]
    with (output_dir / "nodes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(node_rows[0]))
        writer.writeheader()
        writer.writerows(node_rows)

    with (output_dir / "edges.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "source_node_index",
                "target_node_index",
                "source_artwork_id",
                "target_artwork_id",
                "weight",
            ]
        )
        for (source, target), weight in sorted(edge_weights.items()):
            writer.writerow(
                [source, target, node_ids[source], node_ids[target], weight]
            )

    with (output_dir / "group.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for record, artwork_id in zip(selected, node_ids):
            label = _as_text(record[label_field])
            output.write(f"{artwork_id} {label_ids[label]}\n")

    # Dataset currently treats these pairs as the walk-neighborhood matrix W.
    (output_dir / "walks.txt").write_text(
        edge_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    with (output_dir / "label_mapping.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["label_id", label_field])
        writer.writerows((index, label) for label, index in label_ids.items())

    with (output_dir / "artworks.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        fieldnames = [*CSV_FIELDS, "normalized_text"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record, normalized_text in zip(selected, texts):
            row = _csv_row(record)
            row["normalized_text"] = normalized_text
            writer.writerow(row)

    result = {
        "processed_artworks": len(selected),
        "dropped_records": dropped,
        "classes": len(labels),
        "k": len(labels),
        "features": int(features.shape[1]),
        "edges": len(edge_weights),
    }
    label_counts = {
        label: sum(_as_text(record[label_field]) == label for record in selected)
        for label in labels
    }
    artist_counts: dict[str, int] = defaultdict(int)
    for record in selected:
        artist_counts[_as_text(record["artist_id"])] += 1
    metadata = {
        **result,
        "label_field": label_field,
        "feature_fields": list(feature_fields),
        "optional_feature_coverage": feature_coverage,
        "min_optional_feature_coverage": optional_feature_coverage,
        "class_counts": label_counts,
        "class_balance": {
            "min_classes": min_classes,
            "max_classes": max_classes,
            "min_class_samples": min_class_samples,
            "max_class_ratio": max_class_ratio,
        },
        "artist_limits": {
            "min_artworks": min_per_artist,
            "max_artworks": max_per_artist,
            "largest_artist_count": max(artist_counts.values()),
        },
        "edge_endpoints": {
            "source": "source_node_index/source_artwork_id",
            "target": "target_node_index/target_artwork_id",
            "description": "Undirected artwork link between works by the same artist.",
        },
        "feature_max": feature_limit,
        "max_neighbors_per_node": neighbors,
        "graph_strategy": "same_artist_cyclic_neighbors",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl AIC artworks and create CDNMF graph/feature/label files."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "crawl", "preprocess"),
        default="all",
        help="Run the crawl, preprocessing, or both stages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25_000,
        help="Target number of unique artworks with an artist ID and label.",
    )
    parser.add_argument(
        "--processed-limit",
        type=int,
        default=21_000,
        help="Maximum number of raw artworks considered during preprocessing.",
    )
    parser.add_argument(
        "--label-field",
        choices=("artwork_type_title",),
        default="artwork_type_title",
        help="Artwork category used as the evaluation label.",
    )
    parser.add_argument("--feature-limit", type=int, default=128)
    parser.add_argument(
        "--neighbors",
        type=int,
        default=20,
        help="Maximum same-artist neighbors per artwork.",
    )
    parser.add_argument("--max-classes", type=int, default=8)
    parser.add_argument("--min-classes", type=int, default=6)
    parser.add_argument("--min-class-samples", type=int, default=100)
    parser.add_argument("--max-class-ratio", type=float, default=3.0)
    parser.add_argument("--max-per-artist", type=int, default=200)
    parser.add_argument("--min-per-artist", type=int, default=11)
    parser.add_argument("--optional-feature-coverage", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Database") / "artic_25000",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Preprocess an existing raw CSV instead of crawling the API.",
    )
    args = parser.parse_args()
    if args.limit < 2 or args.processed_limit < 2:
        parser.error("--limit and --processed-limit must both be at least 2.")
    if args.feature_limit < 1 or args.neighbors < 2:
        parser.error("--feature-limit must be positive and --neighbors at least 2.")
    if args.min_classes < 2 or args.max_classes < args.min_classes:
        parser.error("--max-classes must be at least --min-classes.")
    if args.min_class_samples < 1:
        parser.error("--min-class-samples must be positive.")
    if args.max_class_ratio <= 1:
        parser.error("--max-class-ratio must be greater than 1.")
    if args.min_per_artist < 1 or args.max_per_artist < args.min_per_artist:
        parser.error("--max-per-artist must be at least --min-per-artist.")
    if not 0 <= args.optional_feature_coverage <= 1:
        parser.error("--optional-feature-coverage must be between 0 and 1.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw_path = args.output_dir / "artworks_raw.csv"
    if args.stage in ("all", "crawl"):
        downloaded = crawl_artworks(args.limit, raw_path)
        LOGGER.info("Saved %s complete raw records to %s", downloaded, raw_path)
    if args.stage == "crawl":
        return
    if args.input_csv:
        raw_path = args.input_csv
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Raw artwork CSV not found: {raw_path}. Run the crawl stage first "
            "or pass --input-csv."
        )
    result = prepare_dataset(
        read_artwork_csv(raw_path, args.limit),
        args.output_dir,
        args.processed_limit,
        args.label_field,
        args.feature_limit,
        args.neighbors,
        args.min_classes,
        args.max_classes,
        args.min_class_samples,
        args.max_class_ratio,
        args.max_per_artist,
        args.min_per_artist,
        args.optional_feature_coverage,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["processed_artworks"] < args.processed_limit:
        LOGGER.warning(
            "Only %s usable records were found (requested up to %s).",
            result["processed_artworks"],
            args.processed_limit,
        )
    LOGGER.warning(
        "CDNMF currently uses dense N×N tensors; the %s-node dataset is exported "
        "for preprocessing, but the existing full-batch model is not practical "
        "to train at this size.",
        result["processed_artworks"],
    )


if __name__ == "__main__":
    main()
