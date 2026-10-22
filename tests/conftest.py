# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared fixtures: every test runs against moto's in-memory S3, never a real account."""

from __future__ import annotations

from typing import Iterator

import boto3
import pytest
from moto import mock_aws

from large_json_context import artifact_store as store_module

BUCKET = "test-artifacts"


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Throwaway credentials and settings, with any real profile removed."""
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("SESSION_STATE_BUCKET", BUCKET)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    store_module.set_artifact_store(None)
    yield
    store_module.set_artifact_store(None)


@pytest.fixture
def s3() -> Iterator[object]:
    """An in-memory S3 with the test bucket created."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client
