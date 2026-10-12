"""Collection uploads stay bounded and resumable; all Hub calls are mocked."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def uploader():
    spec = importlib.util.spec_from_file_location("collection_upload_test", ROOT / "tools" / "upload_model_collection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _safetensors(path, metadata=None):
    header = {"network.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + struct.pack("<f", 1.0))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _blob(path):
    data = path.read_bytes()
    return hashlib.sha1(("blob %d\0" % len(data)).encode("ascii") + data).hexdigest()


@pytest.fixture
def collection(tmp_path, uploader):
    folder = tmp_path / "collection"
    folder.mkdir()
    (folder / "modeling_multistream.py").write_text(uploader._exporter.PUBLIC_LOADER, encoding="utf-8")
    (folder / "README.md").write_text("# Public collection\n", encoding="utf-8")
    (folder / "requirements.txt").write_text("torch==2.2.2\ntorchvision==0.17.2\ntimm==1.0.0\nsafetensors==0.5.0\n", encoding="utf-8")
    entries = []
    for fold in range(2):
        relative = "models/gray_inner/fold_%d" % fold
        directory = folder / relative
        directory.mkdir(parents=True)
        config = {
            "public_format_version": 1,
            "model_config": {"name": "resnet18", "hidden_dim": 8, "freeze_backbones": False, "pretrained": False},
            "streams": ["gray_inner"], "class_to_idx": {"bad": 0, "good": 1},
            "preprocessing": {"image_size": 32, "resize_size": 36, "mean": [0.485, 0.456, 0.406],
                              "std": [0.229, 0.224, 0.225], "resize_backend": "tensor",
                              "interpolation": "bilinear", "antialias": True, "crop": "center"},
            "artifact_kind": "research", "source_format": "legacy_state_dict",
            "settings_provenance": "declared_profiles", "private_training_metadata_omitted": True,
            "legacy_no_split_metadata": True,
        }
        _write_json(directory / "config.json", config)
        _safetensors(directory / "model.safetensors")
        (directory / "README.md").write_text("# Public gray model\n", encoding="utf-8")
        entries.append({"condition": "gray_inner", "fold": fold, "streams": ["gray_inner"], "path": relative,
                        "inference_ready": True, "weights_sha256": _sha(directory / "model.safetensors"),
                        "config_sha256": _sha(directory / "config.json"),
                        "weights_bytes": (directory / "model.safetensors").stat().st_size})
    relative = "historical/legacy_run"
    directory = folder / relative
    directory.mkdir(parents=True)
    _safetensors(directory / "model.safetensors")
    historical = {"historical_format_version": 1, "condition": "historical_legacy_run", "inference_ready": False,
                  "streams": None, "class_to_idx": None, "model_config": None, "preprocessing": None,
                  "source_format": "legacy_raw_state_dict", "private_training_metadata_omitted": True,
                  "reason": "Unknown historical inference settings"}
    _write_json(directory / "config.json", historical)
    (directory / "README.md").write_text("# Historical tensors\n", encoding="utf-8")
    entries.append({"condition": "historical_legacy_run", "fold": None, "streams": None, "path": relative,
                    "inference_ready": False, "weights_sha256": _sha(directory / "model.safetensors"),
                    "config_sha256": _sha(directory / "config.json"), "project": "legacy_run",
                    "source_format": "legacy_raw_state_dict", "reason": historical["reason"],
                    "weights_bytes": (directory / "model.safetensors").stat().st_size})
    catalog = {"catalog_version": 1, "models": entries, "model_count": 3, "inference_ready_count": 2,
               "historical_count": 1, "condition_count": 1, "fold_count": 2,
               "total_weight_bytes": sum(entry["weights_bytes"] for entry in entries),
               "private_training_metadata_omitted": True}
    _write_json(folder / "catalog.json", catalog)
    return folder


class FakeApi:
    def __init__(self):
        self.calls = []
        self.remote = {}
        self.fail_path = None
        self.read_calls = 0
        self.tamper_after_reads = None

    def whoami(self):
        self.calls.append(("identity", {}))
        return {"name": "ntuhophai", "auth": {"accessToken": {"role": "write"}}, "orgs": []}

    def repo_info(self, **kwargs):
        self.calls.append(("info", kwargs))

    def auth_check(self, **kwargs):
        self.calls.append(("auth_check", kwargs))

    def get_paths_info(self, **kwargs):
        self.calls.append(("paths", kwargs))
        self.read_calls += 1
        if self.read_calls == self.tamper_after_reads:
            self.remote["models/gray_inner/fold_0/config.json"].blob_id = "tampered"
        return [self.remote[path] for path in kwargs["paths"] if path in self.remote]

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs))
        if kwargs["path_in_repo"] == self.fail_path:
            raise RuntimeError("A mock failure with a credential that must not be printed: hf_" + "A" * 40)
        directory = Path(kwargs["folder_path"])
        prefix = kwargs["path_in_repo"]
        for name in kwargs["allow_patterns"]:
            file = directory / name
            path = prefix + "/" + name if prefix else name
            self.remote[path] = SimpleNamespace(path=path, size=file.stat().st_size, blob_id=_blob(file),
                                                lfs=SimpleNamespace(sha256=_sha(file)) if name == "model.safetensors" else None)
        return SimpleNamespace(oid="mock-commit")


def _hub(monkeypatch, api):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))


def _uploads(api):
    return [options for kind, options in api.calls if kind == "upload"]


def test_verify_only_has_no_auth_network_or_state(collection, uploader, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Verification must not authenticate or access the Hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=forbidden))
    monkeypatch.setattr(uploader, "_local_login", forbidden)
    result = uploader.upload_collection(collection, verify_only=True)
    assert result["model_count"] == 3 and result["verified_only"] is True
    assert result["uploaded"] == result["skipped"] == 0
    assert not (collection.parent / ".upload-state.json").exists()


@pytest.mark.parametrize("where", ["private_manifest.json", "models/gray_inner/fold_0/private_manifest.json"])
def test_rejects_extra_private_files(collection, uploader, where):
    (collection / where).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        uploader.validate_collection(collection)


@pytest.mark.parametrize("where", ["catalog", "config", "nested_config", "nested_catalog"])
def test_rejects_private_metadata(collection, uploader, where):
    path = collection / ("catalog.json" if "catalog" in where else "models/gray_inner/fold_0/config.json")
    metadata = json.loads(path.read_text())
    target = metadata
    if where == "nested_config":
        target = metadata["model_config"]
    elif where == "nested_catalog":
        target = metadata["models"][0]
    target["private_training_context"] = {"private_identifier": "NEVER_UPLOAD"}
    _write_json(path, metadata)
    with pytest.raises(ValueError):
        uploader.validate_collection(collection)


def test_rejects_nested_values_in_catalog_public_fields(collection, uploader):
    path = collection / "catalog.json"
    metadata = json.loads(path.read_text())
    metadata["models"][0]["project"] = {"private_identifier": "NEVER_UPLOAD"}
    _write_json(path, metadata)
    with pytest.raises(ValueError, match="plain text"):
        uploader.validate_collection(collection)


def test_rejects_links_without_following_them(collection, uploader):
    path = collection / "models/gray_inner/fold_0/README.md"
    target = collection.parent / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("The local account cannot create symlinks")
    with pytest.raises(ValueError, match="Symlinks"):
        uploader.validate_collection(collection)


def test_rejects_modified_loader_and_safetensors_metadata(collection, uploader):
    (collection / "modeling_multistream.py").write_text("raise AssertionError('must not run')", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        uploader.validate_collection(collection)
    (collection / "modeling_multistream.py").write_text(uploader._exporter.PUBLIC_LOADER, encoding="utf-8")
    _safetensors(collection / "models/gray_inner/fold_0/model.safetensors", {"private_identifier": "NEVER_UPLOAD"})
    with pytest.raises(ValueError, match="non-tensor"):
        uploader.validate_collection(collection)


def test_rejects_bad_hashes_counts_and_paths(collection, uploader):
    path = collection / "catalog.json"
    original = json.loads(path.read_text())
    for change in (lambda c: c["models"][0].update(weights_sha256="0" * 64),
                   lambda c: c.update(model_count=999),
                   lambda c: c["models"][0].update(path="../private")):
        metadata = json.loads(json.dumps(original))
        change(metadata)
        _write_json(path, metadata)
        with pytest.raises(ValueError):
            uploader.validate_collection(collection)


def test_underscored_legacy_stream_names_keep_the_declared_order(collection, uploader):
    path = collection / "models/gray_inner/fold_0/config.json"
    config = json.loads(path.read_text())
    config["streams"] = ["_vertical", "_horizontal"]
    _write_json(path, config)
    catalog_path = collection / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["models"][0]["streams"] = config["streams"]
    catalog["models"][0]["config_sha256"] = _sha(path)
    _write_json(catalog_path, catalog)
    _, _, prepared, _ = uploader.validate_collection(collection)
    assert prepared[0]["entry"]["streams"] == ["_vertical", "_horizontal"]


def test_direct_upload_commits_each_model_then_root(collection, uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    result = uploader.upload_collection(collection)
    options = _uploads(api)
    assert [call["path_in_repo"] for call in options] == ["models/gray_inner/fold_0", "models/gray_inner/fold_1",
                                                        "historical/legacy_run", ""]
    assert all(set(call["allow_patterns"]) == uploader.MODEL_FILES for call in options[:-1])
    assert set(options[-1]["allow_patterns"]) == uploader.ROOT_FILES
    assert all(call["create_pr"] is False and "delete_patterns" not in call for call in options)
    assert result["uploaded"] == result["weights_uploaded"] == 3 and result["root_published"] is True
    state = json.loads((collection.parent / ".upload-state.json").read_text())
    assert len(state["entries"]) == 3
    assert all("weights_sha256" in entry and "config_sha256" in entry for entry in state["entries"].values())
    assert not any("state" in name for call in options for name in call["allow_patterns"])


def test_partial_failure_keeps_receipt_and_resume_rechecks_remote(collection, uploader, monkeypatch):
    api = FakeApi()
    api.fail_path = "models/gray_inner/fold_1"
    _hub(monkeypatch, api)
    with pytest.raises(RuntimeError, match="uploaded=1, skipped=0, root_published=false") as failure:
        uploader.upload_collection(collection)
    assert "hf_" not in str(failure.value)
    assert not any(call["path_in_repo"] == "" for call in _uploads(api))
    state = json.loads((collection.parent / ".upload-state.json").read_text())
    assert list(state["entries"]) == ["models/gray_inner/fold_0"]
    api.fail_path = None
    api.calls.clear()
    result = uploader.upload_collection(collection)
    assert result["skipped"] == 1 and result["uploaded"] == 2 and result["root_published"] is True
    assert _uploads(api)[-1]["path_in_repo"] == ""


def test_remote_config_tampering_is_not_hidden_by_local_receipt(collection, uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    uploader.upload_collection(collection)
    api.remote["models/gray_inner/fold_0/config.json"].blob_id = "tampered"
    api.calls.clear()
    result = uploader.upload_collection(collection)
    assert result["uploaded"] == 1 and result["skipped"] == 2
    assert set(_uploads(api)[0]["allow_patterns"]) == uploader.MODEL_FILES


def test_remote_lfs_hash_tampering_forces_a_model_commit(collection, uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    uploader.upload_collection(collection)
    api.remote["models/gray_inner/fold_0/model.safetensors"].lfs.sha256 = "0" * 64
    api.calls.clear()
    result = uploader.upload_collection(collection)
    assert result["weights_uploaded"] == 1 and result["skipped"] == 2


def test_remote_matches_allow_resume_without_a_local_receipt(collection, uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    uploader.upload_collection(collection)
    (collection.parent / ".upload-state.json").unlink()
    api.calls.clear()
    result = uploader.upload_collection(collection)
    assert result["skipped"] == 3 and result["uploaded"] == 0
    assert [call["path_in_repo"] for call in _uploads(api)] == [""]


def test_local_source_change_before_upload_stops_without_catalog(collection, uploader, monkeypatch):
    api = FakeApi()
    original = api.get_paths_info
    def change_source(**kwargs):
        result = original(**kwargs)
        if api.read_calls == 1:
            (collection / "models/gray_inner/fold_0/README.md").write_text("changed after validation", encoding="utf-8")
        return result
    api.get_paths_info = change_source
    _hub(monkeypatch, api)
    with pytest.raises(RuntimeError, match="uploaded=0"):
        uploader.upload_collection(collection)
    assert not _uploads(api)


def test_readme_change_updates_only_the_small_file(collection, uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    uploader.upload_collection(collection)
    (collection / "models/gray_inner/fold_0/README.md").write_text("# Updated code URL\n", encoding="utf-8")
    api.calls.clear()
    result = uploader.upload_collection(collection)
    assert result["uploaded"] == result["readmes_updated"] == 1
    assert result["weights_uploaded"] == 0 and result["skipped"] == 2
    assert _uploads(api)[0]["allow_patterns"] == ["README.md"]


def test_final_remote_tampering_withholds_catalog(collection, uploader, monkeypatch):
    api = FakeApi()
    api.tamper_after_reads = 4
    _hub(monkeypatch, api)
    with pytest.raises(RuntimeError, match="final remote model verification"):
        uploader.upload_collection(collection)
    assert not any(call["path_in_repo"] == "" for call in _uploads(api))


def test_state_must_stay_outside_collection(collection, uploader):
    with pytest.raises(ValueError, match="outside"):
        uploader.upload_collection(collection, verify_only=True, state_file=collection / "state.json")


def test_broken_state_link_is_rejected_without_creating_its_target(collection, uploader):
    state = collection.parent / "linked-state.json"
    target = collection.parent / "never-created.json"
    try:
        state.symlink_to(target)
    except OSError:
        pytest.skip("The local account cannot create symlinks")
    with pytest.raises(ValueError, match="Linked"):
        uploader.upload_collection(collection, verify_only=True, state_file=state)
    assert not target.exists()


def test_linked_collection_ancestor_is_rejected(collection, uploader):
    alias = collection.parent / "linked-parent"
    try:
        alias.symlink_to(collection.parent, target_is_directory=True)
    except OSError:
        pytest.skip("The local account cannot create symlinks")
    with pytest.raises(ValueError, match="Linked"):
        uploader.validate_collection(alias / collection.name)


def test_login_only_checks_write_access_without_upload_or_folder(uploader, monkeypatch):
    api = FakeApi()
    _hub(monkeypatch, api)
    uploader.main(["--login-only"])
    assert not _uploads(api)
    assert any(kind == "auth_check" and kwargs["write"] is True for kind, kwargs in api.calls)


def test_wrong_identity_login_is_rechecked(uploader, monkeypatch):
    api = FakeApi()
    api.whoami = lambda: {"name": "other", "orgs": [], "auth": {"accessToken": {"role": "write"}}}
    _hub(monkeypatch, api)
    logins = []
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(uploader, "_local_login", lambda: logins.append(True))
    with pytest.raises(RuntimeError, match="namespace write access"):
        uploader.ensure_auth()
    assert logins == [True] and not _uploads(api)


def test_token_override_does_not_trigger_login(uploader, monkeypatch):
    api = FakeApi()
    api.whoami = lambda: {"name": "ntuhophai", "auth": {"accessToken": {"role": "read"}}}
    _hub(monkeypatch, api)
    monkeypatch.setenv("HF_TOKEN", "fixture-never-a-real-token")
    monkeypatch.setattr(uploader, "_local_login", lambda: pytest.fail("Must not overwrite token override"))
    with pytest.raises(RuntimeError, match="HF_TOKEN overrides"):
        uploader.ensure_auth()


def test_missing_remote_hash_api_never_trusts_a_filename_listing(collection, uploader, monkeypatch):
    api = FakeApi()
    api.get_paths_info = None
    api.list_repo_files = lambda **kwargs: ["models/gray_inner/fold_0/model.safetensors"]
    _hub(monkeypatch, api)
    with pytest.raises(RuntimeError, match="uploaded=0"):
        uploader.upload_collection(collection)
    assert not _uploads(api)
