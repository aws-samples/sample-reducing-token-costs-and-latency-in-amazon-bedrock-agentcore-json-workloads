# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from large_json_context.artifact_store import artifact_store
from large_json_context.state import TOPTRequest, TOPTResponse, dehydrate_state, hydrate_state

REQUEST = {"tasks": [{"task_id": f"T-{i}"} for i in range(50)]}
RESPONSES = [{"scenario_id": "S1", "assignments": []}, {"scenario_id": "S2", "assignments": []}]


def test_dehydrate_keeps_heavy_payloads_out_of_persisted_state() -> None:
    state = {"messages": [HumanMessage("hi")], "request_raw": REQUEST, "responses_raw": RESPONSES}
    persisted, artifacts = dehydrate_state(state)
    assert set(artifacts) == {"request_ref", "responses_ref"}
    assert "request_raw" not in persisted and "responses_raw" not in persisted
    assert persisted["request_ref"] is None  # filled in by the caller after upload


def test_round_trip_restores_messages_and_domain_objects(s3) -> None:
    state = {
        "messages": [HumanMessage("Which tasks are unassigned?"), AIMessage("Checking.")],
        "request_raw": REQUEST,
        "responses_raw": RESPONSES,
    }
    persisted, artifacts = dehydrate_state(state)
    for ref_field, payload in artifacts.items():
        persisted[ref_field] = artifact_store.put_json(payload)

    restored = hydrate_state(
        persisted,
        request_raw=artifact_store.get_json(persisted["request_ref"]),
        responses_raw=artifact_store.get_json(persisted["responses_ref"]),
    )
    assert [m.content for m in restored["messages"]] == ["Which tasks are unassigned?", "Checking."]
    assert isinstance(restored["messages"][0], HumanMessage)
    assert isinstance(restored["request"], TOPTRequest) and restored["request"].raw == REQUEST
    assert all(isinstance(r, TOPTResponse) for r in restored["responses"])
    assert restored["request_ref"] == persisted["request_ref"]


def test_hydrate_without_payloads_keeps_references_for_later() -> None:
    restored = hydrate_state({"messages": [], "request_ref": "f" * 64})
    assert restored["request"] is None and restored["responses"] == []
    assert restored["request_ref"] == "f" * 64
