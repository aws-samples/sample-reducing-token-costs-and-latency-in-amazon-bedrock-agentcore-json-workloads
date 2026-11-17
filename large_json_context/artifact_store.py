# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Content-addressed artifact store backed by Amazon S3.

Strategy 1 from the post: heavy payloads live in Amazon S3, and session state keeps only a
reference to each one. The reference is the SHA-256 digest of the payload's canonical JSON, so
identical payloads always map to the same object and re-uploading unchanged content is skipped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# The format every reference this store produces must match. See _key.
_REF_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_bytes(payload: Any) -> bytes:
    """Serialize a payload deterministically: sorted keys, compact separators, UTF-8."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _content_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class S3ArtifactStore:
    """Content-addressed artifact store backed by Amazon S3."""

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        # Lazy import: boto3 only required if this backend is actually used.
        import boto3  # noqa: PLC0415

        self.bucket = bucket or os.getenv("SESSION_STATE_BUCKET")
        self.prefix = (prefix or os.getenv("ARTIFACT_STORE_PREFIX", "artifacts")).strip("/")
        # No hardcoded default Region. An explicit argument wins, then
        # AWS_REGION, which boto3 does not read on its own. Otherwise the value is None and
        # boto3 falls back to its default chain: AWS_DEFAULT_REGION, the profile's region,
        # then instance metadata.
        region_name = region or os.getenv("AWS_REGION")
        if not self.bucket:
            raise ValueError("S3ArtifactStore requires SESSION_STATE_BUCKET")

        profile = os.getenv("AWS_PROFILE")
        session = (
            boto3.Session(profile_name=profile, region_name=region_name)
            if profile
            else boto3.Session(region_name=region_name)
        )
        # Fail here rather than let the S3 client fall back to
        # us-east-1 without an error, which puts artifacts in an unexpected Region.
        if session.region_name is None:
            raise ValueError(
                "S3ArtifactStore needs an AWS Region: set AWS_REGION, AWS_DEFAULT_REGION, "
                "or a region in your AWS profile"
            )
        self._s3 = session.client("s3")
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        logger.info("S3ArtifactStore initialized: bucket=%s, prefix=%s", self.bucket, self.prefix)

    def _key(self, ref: str) -> str:
        # References reach this method from the model, through the
        # paging tool, so accept only the digest format this store produces. That stops a
        # crafted value such as "../other/object" from addressing an arbitrary key.
        if not _REF_PATTERN.fullmatch(ref):
            raise ValueError(f"not an artifact reference: {ref!r}")
        return f"{self.prefix}/{ref}.json"

    def put_json(self, payload: Any) -> str:
        ref = _content_hash(payload)
        with self._lock:
            if ref in self._cache:
                # Identical content already uploaded by this process, skip the PUT.
                return ref
        self._s3.put_object(
            Bucket=self.bucket,
            Key=self._key(ref),
            Body=_canonical_bytes(payload),
            ContentType="application/json",
        )
        with self._lock:
            self._cache[ref] = payload
        return ref

    def get_json(self, ref: str) -> Any:
        with self._lock:
            if ref in self._cache:
                return self._cache[ref]
        resp = self._s3.get_object(Bucket=self.bucket, Key=self._key(ref))
        payload = json.loads(resp["Body"].read().decode("utf-8"))
        with self._lock:
            self._cache[ref] = payload
        return payload


# The store is created on first use rather than at import, so importing this module doesn't
# require SESSION_STATE_BUCKET to be set.
_store: Optional[S3ArtifactStore] = None
_store_lock = threading.Lock()


def get_artifact_store() -> S3ArtifactStore:
    """Return the process-wide store, creating it from the environment on first use."""
    global _store
    with _store_lock:
        if _store is None:
            _store = S3ArtifactStore()
        return _store


def set_artifact_store(store: Optional[S3ArtifactStore]) -> None:
    """Replace the process-wide store, or pass None to reset it. Useful in tests."""
    global _store
    with _store_lock:
        _store = store


class _LazyArtifactStore:
    """Stands in for the store and creates the real one the first time it is used."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_artifact_store(), name)


artifact_store = _LazyArtifactStore()
