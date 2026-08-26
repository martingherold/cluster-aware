#!/usr/bin/env python3
"""Download UCI Iris and create two Cluster-Aware Clustering instances.

Each selected observation is used as both a client and a discrete candidate
center. Labels are used only to construct balanced, nested samples and are
excluded from the optimization instances. Distances are Euclidean and are
emitted center-major, as required by the C++ JSON parser.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence
import urllib.error
import urllib.request
import zipfile


IRIS_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/53/iris"
IRIS_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"
IRIS_ARCHIVE_SHA256 = "d11fe30213d36434a0879aab7cb00ce3c812eb7ba2495874438abff7b7b762e9"
IRIS_DATA_MEMBER = "bezdekIris.data"
MAX_ARCHIVE_BYTES = 1_000_000
DOWNLOAD_TIMEOUT_SECONDS = 30.0
NUMBER_OF_CLUSTERS = 3
IRIS_CLASS_COUNTS = {
    "Iris-setosa": 50,
    "Iris-versicolor": 50,
    "Iris-virginica": 50,
}
SAMPLING_SALT = "cluster-aware-iris-stratified-sample-v1"


class PreparationError(RuntimeError):
    """Raised when downloading or transforming the dataset fails."""


@dataclass(frozen=True)
class IrisObservation:
    features: tuple[float, ...]
    label: str


@dataclass(frozen=True)
class InstancePreset:
    filename: str
    samples_per_class: int
    ell: int

    @property
    def observation_count(self) -> int:
        return samples_per_class_to_observation_count(self.samples_per_class)


INSTANCE_PRESETS = (
    InstancePreset("iris-15.json", samples_per_class=5, ell=2),
    InstancePreset("iris-30.json", samples_per_class=10, ell=5),
)


def samples_per_class_to_observation_count(samples_per_class: int) -> int:
    return samples_per_class * len(IRIS_CLASS_COUNTS)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the corrected UCI Iris data and emit 15- and "
            "30-observation Cluster-Aware Clustering instances."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for iris-15.json and iris-30.json",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="z-score each feature using population standard deviation before computing distances",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace preset files that already exist",
    )
    return parser.parse_args(argv)


def download_official_archive() -> bytes:
    request = urllib.request.Request(
        IRIS_ARCHIVE_URL,
        headers={"User-Agent": "cluster-aware-clustering-dataset-preparer/1"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            archive = response.read(MAX_ARCHIVE_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise PreparationError(
            f"could not download the UCI Iris archive from {IRIS_ARCHIVE_URL}: {error}"
        ) from error

    if len(archive) > MAX_ARCHIVE_BYTES:
        raise PreparationError(
            f"downloaded archive exceeds the {MAX_ARCHIVE_BYTES}-byte safety limit"
        )
    return archive


def verify_official_archive(archive: bytes) -> None:
    actual_checksum = hashlib.sha256(archive).hexdigest()
    if actual_checksum != IRIS_ARCHIVE_SHA256:
        raise PreparationError(
            "UCI Iris archive checksum mismatch: "
            f"expected {IRIS_ARCHIVE_SHA256}, got {actual_checksum}. "
            "The upstream archive may have changed; inspect it before updating the pinned checksum."
        )


def parse_iris_csv(text: str) -> list[IrisObservation]:
    observations: list[IrisObservation] = []
    for line_number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 5:
            raise PreparationError(
                f"Iris row {line_number} has {len(row)} fields; expected five"
            )

        try:
            features = tuple(float(value.strip()) for value in row[:4])
        except ValueError as error:
            raise PreparationError(
                f"Iris row {line_number} contains a nonnumeric feature"
            ) from error
        if not all(math.isfinite(value) for value in features):
            raise PreparationError(
                f"Iris row {line_number} contains a non-finite feature"
            )

        label = row[4].strip()
        if not label:
            raise PreparationError(f"Iris row {line_number} has an empty label")
        observations.append(IrisObservation(features, label))

    if not observations:
        raise PreparationError("Iris data file contains no observations")
    return observations


def extract_official_observations(archive: bytes) -> list[IrisObservation]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as dataset_zip:
            raw_data = dataset_zip.read(IRIS_DATA_MEMBER)
    except (KeyError, zipfile.BadZipFile, OSError) as error:
        raise PreparationError(
            f"archive does not contain a readable '{IRIS_DATA_MEMBER}'"
        ) from error

    try:
        observations = parse_iris_csv(raw_data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise PreparationError(f"'{IRIS_DATA_MEMBER}' is not valid UTF-8") from error

    if len(observations) != 150:
        raise PreparationError(
            f"official Iris data should contain 150 observations, got {len(observations)}"
        )
    actual_labels = dict(Counter(observation.label for observation in observations))
    if actual_labels != IRIS_CLASS_COUNTS:
        raise PreparationError(
            "unexpected Iris class distribution: "
            f"expected {IRIS_CLASS_COUNTS}, got {actual_labels}"
        )
    return observations


def deterministic_stratified_indices(
    observations: Sequence[IrisObservation], samples_per_class: int
) -> list[int]:
    """Select a stable, balanced sample whose prefixes form nested datasets."""
    if samples_per_class < 1:
        raise PreparationError("samples per class must be positive")

    indices_by_label: dict[str, list[int]] = {
        label: [] for label in IRIS_CLASS_COUNTS
    }
    for index, observation in enumerate(observations):
        try:
            indices_by_label[observation.label].append(index)
        except KeyError as error:
            raise PreparationError(
                f"unexpected Iris class label '{observation.label}'"
            ) from error

    selected_indices: list[int] = []
    for label, available_indices in indices_by_label.items():
        if len(available_indices) < samples_per_class:
            raise PreparationError(
                f"class '{label}' has {len(available_indices)} observations; "
                f"cannot select {samples_per_class}"
            )

        def stable_rank(index: int) -> bytes:
            sample_key = f"{SAMPLING_SALT}\0{label}\0{index}".encode("utf-8")
            return hashlib.sha256(sample_key).digest()

        selected_indices.extend(
            sorted(available_indices, key=stable_rank)[:samples_per_class]
        )

    return sorted(selected_indices)


def validate_feature_matrix(features: Sequence[Sequence[float]]) -> int:
    if not features:
        raise PreparationError("at least one observation is required")
    dimension = len(features[0])
    if dimension == 0:
        raise PreparationError("observations must have at least one feature")
    for row_index, row in enumerate(features):
        if len(row) != dimension:
            raise PreparationError(
                f"observation {row_index} has dimension {len(row)}; expected {dimension}"
            )
        if not all(math.isfinite(value) for value in row):
            raise PreparationError(
                f"observation {row_index} contains a non-finite feature"
            )
    return dimension


def standardize_features(
    features: Sequence[Sequence[float]],
) -> list[tuple[float, ...]]:
    dimension = validate_feature_matrix(features)
    observation_count = len(features)
    means = [
        math.fsum(row[column] for row in features) / observation_count
        for column in range(dimension)
    ]
    standard_deviations = []
    for column, mean in enumerate(means):
        variance = (
            math.fsum((row[column] - mean) ** 2 for row in features) / observation_count
        )
        standard_deviation = math.sqrt(variance)
        if standard_deviation == 0.0:
            raise PreparationError(f"feature column {column} has zero variance")
        standard_deviations.append(standard_deviation)

    return [
        tuple(
            (row[column] - means[column]) / standard_deviations[column]
            for column in range(dimension)
        )
        for row in features
    ]


def center_major_euclidean_distances(
    features: Sequence[Sequence[float]],
) -> list[list[float]]:
    validate_feature_matrix(features)
    return [[math.dist(center, client) for client in features] for center in features]


def build_instance_document(
    features: Sequence[Sequence[float]],
    *,
    instance_id: str,
    number_of_clusters: int,
    inner_norm: str,
    ell: int | None,
) -> dict[str, Any]:
    validate_feature_matrix(features)
    if not instance_id:
        raise PreparationError("instance id must not be empty")
    if number_of_clusters < 1 or number_of_clusters > len(features):
        raise PreparationError(
            "number of clusters must be between one and the observation count"
        )

    if inner_norm == "top_l":
        if ell is None or ell < 1:
            raise PreparationError("Top-l instances require ell >= 1")
        inner_norm_document = {"type": "top_l", "parameters": {"l": ell}}
    elif inner_norm == "l1":
        if ell is not None:
            raise PreparationError("ell is only valid for Top-l instances")
        inner_norm_document = {"type": "l1", "parameters": {}}
    else:
        raise PreparationError(f"unsupported inner norm '{inner_norm}'")

    return {
        "format": "cluster-aware-instance",
        "version": 1,
        "id": instance_id,
        "number_of_clusters": number_of_clusters,
        "inner_norm": inner_norm_document,
        "outer_norm": {"type": "l1", "parameters": {}},
        "distance_matrix": {
            "client_count": len(features),
            "center_count": len(features),
            "layout": "center_major",
            "values": center_major_euclidean_distances(features),
        },
    }


def write_json_atomically(
    document: dict[str, Any], output_path: Path, force: bool
) -> None:
    if output_path.exists() and not force:
        raise PreparationError(
            f"output file '{output_path}' already exists; pass --force to replace it"
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_path.parent,
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                json.dump(document, output, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except (OSError, ValueError) as error:
        raise PreparationError(f"could not write '{output_path}': {error}") from error


def generated_instance_id(
    *, observation_count: int, ell: int, standardized: bool
) -> str:
    preprocessing = "standardized" if standardized else "raw"
    return (
        f"uci-iris-{preprocessing}-n{observation_count}-"
        f"top-l{ell}-k{NUMBER_OF_CLUSTERS}"
    )


def prepare(args: argparse.Namespace) -> list[tuple[Path, dict[str, Any]]]:
    output_paths = [args.output_dir / preset.filename for preset in INSTANCE_PRESETS]
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not args.force:
        formatted_paths = ", ".join(f"'{path}'" for path in existing_paths)
        raise PreparationError(
            f"output file(s) already exist: {formatted_paths}; "
            "pass --force to replace them"
        )

    archive = download_official_archive()
    verify_official_archive(archive)
    observations = extract_official_observations(archive)
    all_features: Sequence[Sequence[float]] = [
        observation.features for observation in observations
    ]
    if args.standardize:
        all_features = standardize_features(all_features)

    prepared_instances: list[tuple[Path, dict[str, Any]]] = []
    for preset, output_path in zip(INSTANCE_PRESETS, output_paths, strict=True):
        selected_indices = deterministic_stratified_indices(
            observations, preset.samples_per_class
        )
        features = [all_features[index] for index in selected_indices]
        document = build_instance_document(
            features,
            instance_id=generated_instance_id(
                observation_count=preset.observation_count,
                ell=preset.ell,
                standardized=args.standardize,
            ),
            number_of_clusters=NUMBER_OF_CLUSTERS,
            inner_norm="top_l",
            ell=preset.ell,
        )
        prepared_instances.append((output_path, document))
    return prepared_instances


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        prepared_instances = prepare(args)
        for output_path, document in prepared_instances:
            write_json_atomically(document, output_path, args.force)
    except PreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    preprocessing = "standardized" if args.standardize else "raw"
    for output_path, document in prepared_instances:
        observation_count = document["distance_matrix"]["client_count"]
        print(
            f"Wrote {observation_count}-client, {observation_count}-center "
            f"{preprocessing} UCI Iris instance to '{output_path}'."
        )
    print(f"Source: {IRIS_DATASET_PAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
