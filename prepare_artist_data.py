import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def _tag_text(value):
    if pd.isna(value):
        return ""
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return str(value)
    if isinstance(parsed, dict):
        return " ".join(str(key).replace("_", " ") for key in parsed)
    return str(parsed)


def prepare_artist_data(artists_path, edges_path, output_dir):
    artists_path = Path(artists_path)
    edges_path = Path(edges_path)
    output_dir = Path(output_dir)

    artists = pd.read_csv(artists_path, dtype={"artist_id": str})
    edges = pd.read_csv(edges_path, dtype={"source": str, "target": str})
    if "artist_id" not in artists or artists["artist_id"].duplicated().any():
        raise ValueError("artist_csb.csv must contain unique artist_id values.")
    if not {"source", "target", "weight"}.issubset(edges.columns):
        raise ValueError("edges.csv must contain source, target, and weight columns.")
    if edges["weight"].isna().any():
        raise ValueError("Edge weights cannot be missing.")
    edge_weights = edges["weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(edge_weights).all() or (edge_weights < 0).any():
        raise ValueError("Edge weights must be finite, nonnegative numbers.")

    label_column = (
        "department_title" if "department_title" in artists.columns else "main_department"
    )
    if label_column not in artists.columns:
        raise ValueError("artist data must contain department_title or main_department.")
    if artists[label_column].isna().any():
        raise ValueError(f"{label_column} contains missing values.")

    artist_ids = artists["artist_id"].tolist()
    node_index = {artist_id: index for index, artist_id in enumerate(artist_ids)}
    edges = edges.loc[edges["weight"] > 0].copy()
    known = edges["source"].isin(node_index) & edges["target"].isin(node_index)
    dropped_edges = int((~known).sum())
    edges = edges.loc[known & (edges["source"] != edges["target"])]

    rows = edges["source"].map(node_index).to_numpy(dtype=np.int64)
    columns = edges["target"].map(node_index).to_numpy(dtype=np.int64)
    weights = edges["weight"].to_numpy(dtype=np.float32)
    adjacency = sp.coo_matrix(
        (weights, (rows, columns)), shape=(len(artists), len(artists))
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    tag_columns = [
        column
        for column in ("subject_tags", "style_tags", "technique_tags", "material_tags")
        if column in artists.columns
    ]
    documents = []
    for _, artist in artists.iterrows():
        parts = [_tag_text(artist[column]) for column in tag_columns]
        for column in ("main_century", "main_place_of_origin"):
            if column in artists.columns and pd.notna(artist[column]):
                parts.append(f"{column} {artist[column]}")
        documents.append(" ".join(part for part in parts if part))

    text_features = TfidfVectorizer(
        max_features=256,
        token_pattern=r"(?u)\b\w+\b",
    ).fit_transform(documents).astype(np.float32)
    artwork_count = pd.to_numeric(
        artists.get("artwork_count", pd.Series(0, index=artists.index)),
        errors="coerce",
    ).fillna(0).to_numpy(dtype=np.float32)
    artwork_count = np.log1p(np.maximum(artwork_count, 0))
    if artwork_count.max() > 0:
        artwork_count /= artwork_count.max()
    features = sp.hstack(
        [text_features, sp.csr_matrix(artwork_count.reshape(-1, 1))],
        format="csr",
    ).toarray()

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "features.txt").open("w", encoding="utf-8") as file:
        for artist_id, vector in zip(artist_ids, features):
            values = " ".join(f"{value:.8g}" for value in vector)
            file.write(f"{artist_id} {values}\n")

    labels = artists[label_column].astype(str)
    label_codes, label_names = pd.factorize(labels, sort=True)
    with (output_dir / "group.txt").open("w", encoding="utf-8") as file:
        for artist_id, label in zip(artist_ids, label_codes):
            file.write(f"{artist_id} {label}\n")
    (output_dir / "label_mapping.json").write_text(
        json.dumps(
            {int(code): str(name) for code, name in enumerate(label_names)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    coo = adjacency.tocoo()
    with (output_dir / "edges.txt").open("w", encoding="utf-8") as file:
        for source, target, weight in zip(coo.row, coo.col, coo.data):
            if source < target:
                file.write(
                    f"{artist_ids[source]} {artist_ids[target]} {weight:.8g}\n"
                )
    (output_dir / "walks.txt").write_text("", encoding="utf-8")

    return {
        "nodes": len(artists),
        "edges": adjacency.nnz // 2,
        "features": features.shape[1],
        "classes": len(label_names),
        "label_column": label_column,
        "dropped_edges": dropped_edges,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert artist_csb.csv and edges.csv to CDNMF Database/artist files."
    )
    root = Path(__file__).resolve().parent
    parser.add_argument("--artists", type=Path, default=root / "artist_csb.csv")
    parser.add_argument("--edges", type=Path, default=root / "edges.csv")
    parser.add_argument("--output", type=Path, default=root / "Database" / "artist")
    args = parser.parse_args()

    summary = prepare_artist_data(args.artists, args.edges, args.output)
    print(
        "Generated {nodes} artists, {edges} edges, {features} features, "
        "{classes} departments in {output_dir}; dropped {dropped_edges} "
        "edges with unknown artists. Labels: {label_column}.".format(**summary)
    )


if __name__ == "__main__":
    main()
