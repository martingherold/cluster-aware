#!/usr/bin/env python3

from __future__ import annotations

import contextlib
from collections import Counter
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import prepare_uci_iris as iris  # noqa: E402


def make_official_sized_observations() -> list[iris.IrisObservation]:
    observations = []
    for class_index, label in enumerate(iris.IRIS_CLASS_COUNTS):
        for sample_index in range(50):
            value = float(class_index * 100 + sample_index)
            observations.append(
                iris.IrisObservation(
                    (value, value + 0.25, value + 0.5, value + 0.75), label
                )
            )
    return observations


class PrepareUciIrisTest(unittest.TestCase):
    def test_default_cli_pipeline_without_network(self) -> None:
        observations = make_official_sized_observations()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "iris"
            with (
                mock.patch.object(
                    iris, "download_official_archive", return_value=b"archive"
                ) as download,
                mock.patch.object(iris, "verify_official_archive") as verify,
                mock.patch.object(
                    iris, "extract_official_observations", return_value=observations
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                return_code = iris.main(["--output-dir", str(output_directory)])

            self.assertEqual(return_code, 0)
            download.assert_called_once_with()
            verify.assert_called_once_with(b"archive")
            expected_presets = {
                "iris-15.json": (15, 2),
                "iris-30.json": (30, 5),
            }
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                set(expected_presets),
            )
            for filename, (observation_count, ell) in expected_presets.items():
                document = json.loads(
                    (output_directory / filename).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    document["id"],
                    f"uci-iris-raw-n{observation_count}-top-l{ell}-k3",
                )
                self.assertEqual(document["number_of_clusters"], 3)
                self.assertEqual(
                    document["inner_norm"],
                    {"type": "top_l", "parameters": {"l": ell}},
                )
                self.assertEqual(
                    document["distance_matrix"]["client_count"], observation_count
                )
                self.assertEqual(
                    document["distance_matrix"]["center_count"], observation_count
                )

    def test_stratified_samples_are_balanced_deterministic_and_nested(self) -> None:
        observations = make_official_sized_observations()
        samples = {
            samples_per_class: iris.deterministic_stratified_indices(
                observations, samples_per_class
            )
            for samples_per_class in (5, 10)
        }

        self.assertEqual(
            samples[5], iris.deterministic_stratified_indices(observations, 5)
        )
        self.assertLess(set(samples[5]), set(samples[10]))
        for samples_per_class, indices in samples.items():
            self.assertEqual(indices, sorted(indices))
            self.assertEqual(
                Counter(observations[index].label for index in indices),
                Counter(
                    {label: samples_per_class for label in iris.IRIS_CLASS_COUNTS}
                ),
            )

        with self.assertRaises(iris.PreparationError):
            iris.deterministic_stratified_indices(observations, 51)

    def test_existing_preset_is_rejected_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "iris"
            output_directory.mkdir()
            existing_path = output_directory / "iris-30.json"
            existing_path.write_text("keep me\n", encoding="utf-8")

            with (
                mock.patch.object(iris, "download_official_archive") as download,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                return_code = iris.main(["--output-dir", str(output_directory)])

            self.assertEqual(return_code, 1)
            download.assert_not_called()
            self.assertEqual(
                existing_path.read_text(encoding="utf-8"), "keep me\n"
            )
            self.assertFalse((output_directory / "iris-15.json").exists())

    def test_parse_csv(self) -> None:
        observations = iris.parse_iris_csv(
            "5.1,3.5,1.4,0.2,Iris-setosa\n" "\n" "7.0,3.2,4.7,1.4,Iris-versicolor\n"
        )

        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0].features, (5.1, 3.5, 1.4, 0.2))
        self.assertEqual(observations[1].label, "Iris-versicolor")

        with self.assertRaises(iris.PreparationError):
            iris.parse_iris_csv("5.1,3.5,1.4,Iris-setosa\n")
        with self.assertRaises(iris.PreparationError):
            iris.parse_iris_csv("5.1,nan,1.4,0.2,Iris-setosa\n")

    def test_official_archive_structure_is_validated(self) -> None:
        rows = []
        for label in ("Iris-setosa", "Iris-versicolor", "Iris-virginica"):
            rows.extend(f"1.0,2.0,3.0,4.0,{label}\n" for _ in range(50))

        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr(iris.IRIS_DATA_MEMBER, "".join(rows))

        observations = iris.extract_official_observations(archive_buffer.getvalue())
        self.assertEqual(len(observations), 150)

        with self.assertRaises(iris.PreparationError):
            iris.extract_official_observations(b"not a zip archive")

    def test_standardization_uses_population_statistics(self) -> None:
        standardized = iris.standardize_features([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
        for column in range(2):
            values = [row[column] for row in standardized]
            self.assertAlmostEqual(math.fsum(values) / len(values), 0.0)
            variance = math.fsum(value * value for value in values) / len(values)
            self.assertAlmostEqual(variance, 1.0)

        with self.assertRaises(iris.PreparationError):
            iris.standardize_features([(1.0, 2.0), (1.0, 3.0)])

    def test_instance_is_center_major_and_schema_compatible(self) -> None:
        document = iris.build_instance_document(
            [(0.0, 0.0), (3.0, 4.0), (0.0, 4.0)],
            instance_id="fixture",
            number_of_clusters=2,
            inner_norm="top_l",
            ell=1,
        )

        self.assertEqual(document["format"], "cluster-aware-instance")
        self.assertEqual(document["version"], 1)
        self.assertEqual(
            document["inner_norm"], {"type": "top_l", "parameters": {"l": 1}}
        )
        self.assertEqual(document["outer_norm"], {"type": "l1", "parameters": {}})
        self.assertEqual(
            document["distance_matrix"],
            {
                "client_count": 3,
                "center_count": 3,
                "layout": "center_major",
                "values": [
                    [0.0, 5.0, 4.0],
                    [5.0, 0.0, 3.0],
                    [4.0, 3.0, 0.0],
                ],
            },
        )

    def test_l1_document_rejects_ell(self) -> None:
        points = [(0.0,), (1.0,)]
        document = iris.build_instance_document(
            points,
            instance_id="l1-fixture",
            number_of_clusters=1,
            inner_norm="l1",
            ell=None,
        )
        self.assertEqual(document["inner_norm"], {"type": "l1", "parameters": {}})

        with self.assertRaises(iris.PreparationError):
            iris.build_instance_document(
                points,
                instance_id="invalid",
                number_of_clusters=1,
                inner_norm="l1",
                ell=1,
            )

    def test_checksum_and_atomic_output_guards(self) -> None:
        with self.assertRaises(iris.PreparationError):
            iris.verify_official_archive(b"changed archive")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "instance.json"
            document = {"value": 1}
            iris.write_json_atomically(document, output_path, force=False)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")), document
            )

            with self.assertRaises(iris.PreparationError):
                iris.write_json_atomically({"value": 2}, output_path, force=False)
            iris.write_json_atomically({"value": 2}, output_path, force=True)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")), {"value": 2}
            )


if __name__ == "__main__":
    unittest.main()
