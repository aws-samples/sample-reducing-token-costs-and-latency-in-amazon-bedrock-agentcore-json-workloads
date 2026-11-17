# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Keeping oversized tool results out of the model context.

Strategy 2 from the post pages large lists: the model gets a count, a sample, and a handle, and
fetches further slices on demand. Strategy 3 compacts large dictionaries into a descriptor of
their structure. In both cases the full result is stored in Amazon S3 behind the handle.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .artifact_store import artifact_store

# Lists longer than this are stored and returned as this many items plus a handle.
_LLM_SAMPLE_SIZE = 5
# Serialized results longer than this are replaced by a descriptor. 7,600 characters is about
# 4,000 tokens of dense JSON, which tokenizes at roughly 1.9 characters per token.
_MAX_TOOL_CONTENT_CHARS = 7_600


def compact_list_result(result: List[Any]) -> str:
    """Return a short list in full, or store a long one and return a sample plus a handle."""
    if len(result) > _LLM_SAMPLE_SIZE:
        handle = artifact_store.put_json(result)
        return json.dumps(
            {
                "_oversized": True,
                "_result_handle": handle,
                "total": len(result),
                "sample": result[:_LLM_SAMPLE_SIZE],
            },
            default=str,
        )
    return json.dumps(result, default=str)


def get_result_page_tool(handle: str, field: Optional[str] = None, offset: int = 0, limit: int = 50):
    payload = artifact_store.get_json(handle)
    if isinstance(payload, list):
        return {
            "handle": handle,
            "offset": offset,
            "limit": limit,
            "total": len(payload),
            "items": payload[offset: offset + limit],
        }


def _oversized_descriptor(payload: Dict[str, Any], handle: str) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list):
            fields[key] = {"type": "list", "count": len(value), "sample": value[:5]}
        elif isinstance(value, dict):
            fields[key] = {"type": "dict", "keys": len(value)}
        else:
            fields[key] = {"type": "scalar", "value": value}
    return {"_oversized": True, "_result_handle": handle, "fields": fields}


def compact_tool_result(
    llm_view: Dict[str, Any],
    full_view: Optional[Dict[str, Any]] = None,
    handle: Optional[str] = None,
) -> str:
    """Serialize a dictionary result, replacing it with a descriptor if it is too large.

    `llm_view` is what the model would see, `full_view` is the complete result to store (the
    same as `llm_view` when omitted), and `handle` is an existing handle if the full result has
    already been stored.
    """
    if full_view is None:
        full_view = llm_view
    content = json.dumps(llm_view, default=str)
    if len(content) > _MAX_TOOL_CONTENT_CHARS:
        if handle is None:
            handle = artifact_store.put_json(full_view)
        content = json.dumps(_oversized_descriptor(full_view, handle), default=str)
    return content
