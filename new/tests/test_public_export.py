"""Public export regression tests use generated local weights and images only."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
import numpy as np
from PIL import Image
import yaml

from multistream.data import CaseDataset
from multistream.models import build_model


EXPORT_PATH = Path(__file__).resolve().parents[1] / "tools" / "export_public_model.py"
SPEC = importlib.util.spec_from_file_location("public_export_under_test", EXPORT_PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)

MODEL_CONFIG = {"name": "resnet18", "pretrained": False, "hidden_dim": 8,
                "freeze_backbones": False}
PREPROCESSING = {"image_size": 32, "resize_size": 48,
                 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
CLASSES = {"bad": 0, "good": 1}
PUBLIC_FILES = {"model.safetensors", "config.json", "modeling_multistream.py",
                "README.md", "requirements.txt"}


def _load_public_module(bundle):
    spec = importlib.util.spec_from_file_location(
        "standalone_loader_under_test", bundle / "modeling_multistream.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(model, streams):
    return {
        "format_version": 1, "state_dict": model.state_dict(),
        "model_config": {
            **MODEL_CONFIG,
            "initialization": {streams[0]: "PRIVATE_CHECKPOINT_ORIGIN"},
            "initialization_context": {"train_patient_ids": ["PRIVATE_ID_MARKER"]},
        },
        "streams": list(streams), "class_to_idx": dict(CLASSES),
        "preprocessing": {**PREPROCESSING, "private_notes": "PRIVATE_PREPROCESS_MARKER"},
        "epoch": 1, "fold": 0,
        "train_patient_ids": ["PRIVATE_ID_MARKER"],
        "validation_patient_ids": ["PRIVATE_VALIDATION_MARKER"],
        "test_patient_ids": ["PRIVATE_TEST_MARKER"],
        "train_sample_ids": ["PRIVATE_SAMPLE_MARKER"],
        "validation_sample_ids": ["PRIVATE_VALIDATION_SAMPLE"],
        "test_sample_ids": ["PRIVATE_TEST_SAMPLE"],
        "manifest_fingerprint": "PRIVATE_MANIFEST_MARKER",
        "split_fingerprint": "PRIVATE_SPLIT_MARKER",
        "legacy_source": "PRIVATE_LEGACY_PATH",
        "config": {"data": {"root": "PRIVATE_DATASET_PATH"},
                   "experiment": {"output_dir": "PRIVATE_OUTPUT_PATH"}},
    }


def _legacy_state(model):
    result = {}
    for key, value in model.state_dict().items():
        if key.startswith("backbones."):
            _, index, remainder = key.split(".", 2)
            key = "b%d.%s" % (int(index) + 1, remainder)
        elif key.startswith("fusion."):
            key = "merger." + key[len("fusion."):]
        elif key.startswith("classifier."):
            key = "fc." + key[len("classifier."):]
        result[key] = value
    return result


class PublicExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(412)
        cls.single = build_model(MODEL_CONFIG, ["_vertical"], 2).eval()
        cls.fusion = build_model(MODEL_CONFIG, ["_vertical", "color_full"], 2).eval()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_new_single_and_fusion_roundtrip_and_private_metadata_removed(self):
        for model in (self.single, self.fusion):
            with self.subTest(streams=len(model.streams)), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                source = directory / "source.pth"
                bundle = directory / "public"
                torch.save(_source(model, model.streams), source)
                metadata = exporter.export_public_model(source, bundle)
                self.assertEqual({p.name for p in bundle.iterdir()}, PUBLIC_FILES)
                self.assertEqual(set(metadata["model_config"]),
                                 {"name", "hidden_dim", "freeze_backbones", "pretrained"})
                self.assertFalse(metadata["model_config"]["pretrained"])
                self.assertEqual(metadata["artifact_kind"], "research")  # Epoch 1 is not a demo rule.
                for filename in PUBLIC_FILES - {"model.safetensors"}:
                    self.assertNotIn("PRIVATE_", (bundle / filename).read_text(encoding="utf-8"))
                loader = _load_public_module(bundle)
                restored = loader.load_public_model(bundle)
                self.assertEqual(set(restored.state_dict()), set(model.state_dict()))
                self.assertIn(
                    "bn1.num_batches_tracked" if len(model.streams) == 1
                    else "backbones.0.bn1.num_batches_tracked",
                    restored.state_dict(),
                )
                images = torch.randn(2, len(model.streams), 3, 32, 32)
                with torch.no_grad():
                    torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)

    def test_yaml_extends_relative_paths_and_class_mapping_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            child_dir = directory / "variants"
            child_dir.mkdir()
            base = {
                "data": {"streams": ["old"], "class_to_idx": {"old-negative": 0, "old-positive": 1}},
                "model": MODEL_CONFIG, "preprocessing": PREPROCESSING,
                "experiment": {"output_dir": "PRIVATE_CONFIG_PATH"},
            }
            (directory / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
            child = {"extends": "../base.yaml",
                     "data": {"streams": ["_vertical", "color_full"], "class_to_idx": CLASSES}}
            path = child_dir / "fusion.yaml"
            path.write_text(yaml.safe_dump(child), encoding="utf-8")
            resolved = exporter.resolve_legacy_config(path)
            self.assertEqual(resolved["data"]["class_to_idx"], CLASSES)
            self.assertEqual(resolved["model"]["name"], "resnet18")
            self.assertEqual(resolved["preprocessing"], PREPROCESSING)

    def test_yaml_extends_cycle_and_depth_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
            (directory / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cycle"):
                exporter.resolve_legacy_config(directory / "a.yaml")
            for index in range(4):
                content = ("extends: chain%d.yaml\n" % (index + 1)) if index < 3 else "model: {}\n"
                (directory / ("chain%d.yaml" % index)).write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "depth"):
                exporter.resolve_legacy_config(directory / "chain0.yaml", max_depth=3)

    def test_raw_legacy_requires_config_and_extends_config_exports_strictly(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "legacy.pth"
            torch.save(_legacy_state(self.fusion), source)
            with self.assertRaisesRegex(ValueError, "legacy-config"):
                exporter.export_public_model(source, directory / "no-config")
            base = {"model": MODEL_CONFIG,
                    "preprocessing": {**PREPROCESSING, "resize_backend": "tensor", "antialias": False},
                    "data": {"class_to_idx": CLASSES}}
            (directory / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
            config_path = directory / "fusion.yaml"
            config_path.write_text(
                "extends: base.yaml\ndata:\n  streams: [_vertical, color_full]\n",
                encoding="utf-8",
            )
            bundle = directory / "public"
            metadata = exporter.export_public_model(source, bundle, config_path)
            self.assertTrue(metadata["legacy_no_split_metadata"])
            self.assertEqual(metadata["settings_provenance"], "operator_supplied")
            self.assertEqual(metadata["preprocessing"]["resize_backend"], "tensor")
            self.assertFalse(metadata["preprocessing"]["antialias"])
            restored = _load_public_module(bundle).load_public_model(bundle)
            images = torch.randn(1, 2, 3, 32, 32)
            with torch.no_grad():
                torch.testing.assert_close(self.fusion(images), restored(images), rtol=0, atol=0)
            card = (bundle / "README.md").read_text(encoding="utf-8")
            self.assertIn("case-level random folds", card)
            self.assertIn("operator", card)

    def test_eighth_legacy_branch_and_missing_batchnorm_buffer(self):
        translated = exporter._legacy_keys(
            {"b1.bn1.num_batches_tracked": torch.tensor(3),
             "b8.bn1.num_batches_tracked": torch.tensor(7),
             "merger.2.bias": torch.zeros(8), "fc.bias": torch.zeros(2)}, 8,
        )
        self.assertEqual(set(translated),
                         {"backbones.0.bn1.num_batches_tracked",
                          "backbones.7.bn1.num_batches_tracked",
                          "fusion.2.bias", "classifier.bias"})
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "missing-buffer.pth"
            bad = _source(self.single, self.single.streams)
            del bad["state_dict"]["bn1.num_batches_tracked"]
            torch.save(bad, source)
            output = Path(temp) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "strictly match"):
                exporter.export_public_model(source, output)
            self.assertFalse(output.exists())

    def test_converted_legacy_is_not_remapped_again(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "converted.pth"
            converted = _source(self.fusion, self.fusion.streams)
            converted["legacy_no_split_metadata"] = True
            converted["fold"] = None
            converted["train_patient_ids"] = []
            converted["validation_patient_ids"] = []
            converted["test_patient_ids"] = []
            torch.save(converted, source)
            output = Path(temp) / "public"
            metadata = exporter.export_public_model(source, output)
            self.assertEqual(metadata["source_format"], "converted_legacy")
            self.assertEqual(metadata["preprocessing"]["resize_backend"], "pil")
            self.assertTrue(metadata["preprocessing"]["antialias"])
            restored = _load_public_module(output).load_public_model(output)
            self.assertEqual(set(restored.state_dict()), set(self.fusion.state_dict()))

    def test_smoke_path_forces_demo_and_preview_needs_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "SMOKE_OUTPUT"
            directory.mkdir()
            source = directory / "source.pth"
            torch.save(_source(self.single, self.single.streams), source)
            metadata = exporter.export_public_model(source, None, kind="research", preview=True)
            self.assertEqual(metadata["artifact_kind"], "demo")
            self.assertEqual({p.name for p in directory.iterdir()}, {"source.pth"})

    def test_public_pil_preprocessing_matches_new_case_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image_path = directory / "image.png"
            Image.new("RGB", (60, 48), (55, 120, 200)).save(image_path)
            sample = {"sample_id": "synthetic-case", "patient_id": "synthetic-group",
                      "label": 0, "images": {"_vertical": str(image_path)}}
            expected = CaseDataset([sample], ["_vertical"], PREPROCESSING, training=False)[0]["images"]
            loader = exporter._reference_loader()
            preprocessing = exporter._public_preprocessing(PREPROCESSING, legacy=False)
            actual = loader["preprocess_streams"]([image_path], preprocessing)[0]
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_raw_legacy_requires_explicit_backend_and_antialias(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "legacy.pth"
            torch.save(_legacy_state(self.fusion), source)
            config = {"model": MODEL_CONFIG, "preprocessing": PREPROCESSING,
                      "data": {"streams": list(self.fusion.streams), "class_to_idx": CLASSES}}
            config_path = directory / "profile.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicit resize_backend and antialias"):
                exporter.export_public_model(source, directory / "missing-both", config_path)
            config["preprocessing"] = {**PREPROCESSING, "resize_backend": "tensor"}
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicit resize_backend and antialias"):
                exporter.export_public_model(source, directory / "missing-antialias", config_path)

    def test_public_tensor_preprocessing_respects_both_antialias_versions(self):
        from torchvision.transforms import functional as F
        from torchvision.transforms import InterpolationMode
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pattern.png"
            pixels = np.random.default_rng(19).integers(0, 256, (81, 107, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            loader = exporter._reference_loader()
            results = []
            for antialias in (False, True):
                preprocessing = exporter._public_preprocessing(
                    {**PREPROCESSING, "resize_backend": "tensor", "antialias": antialias}, True,
                )
                with Image.open(path) as image:
                    expected = F.resize(
                        F.to_tensor(image.convert("RGB")), PREPROCESSING["resize_size"],
                        interpolation=InterpolationMode.BILINEAR, antialias=antialias,
                    )
                expected = F.center_crop(expected, [PREPROCESSING["image_size"]] * 2)
                expected = F.normalize(expected, PREPROCESSING["mean"], PREPROCESSING["std"])
                actual = loader["preprocess_streams"]([path], preprocessing)[0, 0]
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                results.append(actual)
            self.assertFalse(torch.allclose(results[0], results[1]))


if __name__ == "__main__":
    unittest.main()
