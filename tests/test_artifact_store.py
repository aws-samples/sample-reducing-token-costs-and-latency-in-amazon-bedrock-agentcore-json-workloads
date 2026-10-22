# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from __future__ import annotations

import pytest

from large_json_context.artifact_store import (
    S3ArtifactStore,
    _canonical_bytes,
    _content_hash,
    artifact_store,
)
from tests.conftest import BUCKET


def test_identical_payloads_share_a_reference() -> None:
    assert _content_hash({"b": 1, "a": [1, 2]}) == _content_hash({"a": [1, 2], "b": 1})
    assert _canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_round_trip_through_s3(s3) -> None:
    store = S3ArtifactStore()
    ref = store.put_json({"rows": list(range(100))})
    assert len(ref) == 64
    obj = s3.get_object(Bucket=BUCKET, Key=f"artifacts/{ref}.json")
    assert obj["ContentType"] == "application/json"
    # A fresh store has an empty cache, so this read really goes to S3.
    assert S3ArtifactStore().get_json(ref) == {"rows": list(range(100))}


def test_unchanged_payload_is_not_uploaded_twice(s3) -> None:
    store = S3ArtifactStore()
    first = store.put_json({"k": "v"})
    s3.delete_object(Bucket=BUCKET, Key=f"artifacts/{first}.json")
    # Served from the in-process cache, so no second PUT recreates the object.
    assert store.put_json({"k": "v"}) == first
    assert "Contents" not in s3.list_objects_v2(Bucket=BUCKET)


@pytest.mark.parametrize("bad", ["../secrets", "abc", "A" * 64, "x" * 64, "0" * 63, ""])
def test_malformed_references_are_rejected(s3, bad: str) -> None:
    with pytest.raises(ValueError, match="not an artifact reference"):
        S3ArtifactStore().get_json(bad)


def test_missing_bucket_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESSION_STATE_BUCKET")
    with pytest.raises(ValueError, match="SESSION_STATE_BUCKET"):
        S3ArtifactStore()


def test_missing_region_fails_instead_of_defaulting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_REGION")
    with pytest.raises(ValueError, match="needs an AWS Region"):
        S3ArtifactStore()


def test_aws_default_region_is_still_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_REGION")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    assert S3ArtifactStore()._s3.meta.region_name == "eu-central-1"


def test_explicit_region_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    assert S3ArtifactStore(region="ap-southeast-2")._s3.meta.region_name == "ap-southeast-2"


def test_module_import_does_not_need_aws(monkeypatch: pytest.MonkeyPatch) -> None:
    # The lazy store means importing never touches configuration; only first use does.
    monkeypatch.delenv("SESSION_STATE_BUCKET")
    with pytest.raises(ValueError, match="SESSION_STATE_BUCKET"):
        artifact_store.put_json({})
