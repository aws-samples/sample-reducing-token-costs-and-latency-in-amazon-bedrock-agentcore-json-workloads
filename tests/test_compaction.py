# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from __future__ import annotations

import json

from large_json_context.compaction import (
    _LLM_SAMPLE_SIZE,
    _MAX_TOOL_CONTENT_CHARS,
    _oversized_descriptor,
    compact_list_result,
    compact_tool_result,
    get_result_page_tool,
)


def test_short_list_passes_through_unchanged(s3) -> None:
    rows = [{"id": i} for i in range(_LLM_SAMPLE_SIZE)]
    assert json.loads(compact_list_result(rows)) == rows


def test_long_list_becomes_sample_and_handle(s3) -> None:
    rows = [{"id": i} for i in range(1_000)]
    compact = json.loads(compact_list_result(rows))
    assert compact["_oversized"] is True
    assert compact["total"] == 1_000
    assert compact["sample"] == rows[:_LLM_SAMPLE_SIZE]

    page = get_result_page_tool(compact["_result_handle"], offset=100, limit=25)
    assert page["items"] == rows[100:125]
    assert page["total"] == 1_000


def test_small_dictionary_passes_through(s3) -> None:
    small = {"scenario_id": "S1", "count": 3}
    assert json.loads(compact_tool_result(small)) == small


def test_large_dictionary_becomes_descriptor(s3) -> None:
    big = {
        "scenario_id": "S1",
        "rows": [{"id": i, "note": "x" * 20} for i in range(500)],
        "by_key": {f"k{i}": i for i in range(500)},
        "_internal": "skipped",
    }
    content = compact_tool_result(big)
    assert len(content) < _MAX_TOOL_CONTENT_CHARS < len(json.dumps(big))
    descriptor = json.loads(content)
    assert descriptor["fields"]["rows"] == {"type": "list", "count": 500, "sample": big["rows"][:5]}
    assert descriptor["fields"]["by_key"] == {"type": "dict", "keys": 500}
    assert descriptor["fields"]["scenario_id"] == {"type": "scalar", "value": "S1"}
    assert "_internal" not in descriptor["fields"]

    # The full payload stays reachable behind the handle.
    from large_json_context.artifact_store import artifact_store

    assert artifact_store.get_json(descriptor["_result_handle"]) == big


def test_existing_handle_is_reused(s3) -> None:
    big = {"rows": list(range(5_000))}
    descriptor = json.loads(compact_tool_result(big, handle="a" * 64))
    assert descriptor["_result_handle"] == "a" * 64


def test_descriptor_shape() -> None:
    assert _oversized_descriptor({"n": 1}, "h") == {
        "_oversized": True,
        "_result_handle": "h",
        "fields": {"n": {"type": "scalar", "value": 1}},
    }
