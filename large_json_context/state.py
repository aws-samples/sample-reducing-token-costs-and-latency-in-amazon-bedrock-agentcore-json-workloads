# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Agent state, its persisted form, and the conversions between them (Strategy 1).

The live graph state holds parsed domain objects for the tools. The persisted state, written to
Amazon Bedrock AgentCore Memory, keeps only messages and references to artifacts stored in
Amazon S3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.load import dumpd, load
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# TOPT stands for Training Optimizer. TOPTRequest and TOPTResponse are the agent's typed domain
# objects: the optimizer's request JSON and response JSON, parsed so tools can work with them.
# The post uses both without defining them, because their fields depend on your optimizer. These
# minimal stand-ins keep the raw dictionary so the code runs. Replace them with your own classes,
# keeping `from_dict`.
@dataclass
class TOPTRequest:
    """Parsed optimizer request. Minimal stand-in; see the note above."""

    raw: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TOPTRequest:
        return cls(raw=data)


@dataclass
class TOPTResponse:
    """One parsed optimizer response. Minimal stand-in; see the note above."""

    raw: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TOPTResponse:
        return cls(raw=data)


# The state as first written, before Strategy 1, with every heavy payload inline. The post shows
# it under the name GraphState; it is renamed here so it can sit beside the updated version.
# Differs from the post: the docstring is indented. In the post it starts at column 0, which
# Python rejects with an IndentationError.
class OriginalGraphState(TypedDict):
    """LangGraph state that combines messages with application data."""
    # Messages from conversation (managed by add_messages reducer)
    messages: Annotated[List[BaseMessage], add_messages]
    # Application-specific state - parsed domain objects
    request: Optional[TOPTRequest]
    responses: List[TOPTResponse]
    validation_report: Optional[Dict[str, Any]]
    basic_stats: Optional[Dict[str, Any]]
    cache_data: Optional[Dict[str, Any]]


class PersistedState(TypedDict, total=False):
    """Serialized session state for AgentCore Memory.
    Heavy fields (request, responses) are stored in S3 and referenced by key.
    Lightweight fields are stored inline.
    """
    messages: List[Any]                       # LangChain-serialized messages
    request_ref: Optional[str]                # S3 key for raw request JSON
    responses_ref: Optional[str]              # S3 key for raw responses JSON list
    cache_data_ref: Optional[str]             # Artifact ref for cache_data payload
    validation_report_ref: Optional[str]      # Artifact ref for validation_report payload
    basic_stats_ref: Optional[str]            # Artifact ref for basic_stats payload


class GraphState(TypedDict):
    """LangGraph state that combines messages with application data."""
    # Messages from conversation (managed by add_messages reducer)
    messages: Annotated[List[BaseMessage], add_messages]

    # Application-specific state - parsed domain objects
    request: Optional[TOPTRequest]
    responses: List[TOPTResponse]
    validation_report: Optional[Dict[str, Any]]
    basic_stats: Optional[Dict[str, Any]]
    cache_data: Optional[Dict[str, Any]]  # Serializable cache data for rich queries
    request_ref: Optional[str]  # S3 key reference for raw request JSON (for persistence)
    responses_ref: Optional[str]  # S3 key reference for raw responses JSON list
    cache_data_ref: Optional[str]  # Artifact ref for cache_data payload
    validation_report_ref: Optional[str]  # Artifact ref for validation_report payload
    basic_stats_ref: Optional[str]  # Artifact ref for basic_stats payload

    # Differs from the post: declared here because dehydrate_state reads these two fields and
    # hydrate_state writes them, but the post's GraphState does not list them.
    request_raw: Optional[Dict[str, Any]]  # Raw request JSON, when loaded this session
    responses_raw: Optional[List[Dict[str, Any]]]  # Raw responses JSON list, when loaded


def dehydrate_state(state: GraphState) -> Tuple[PersistedState, Dict[str, Any]]:
    persisted: PersistedState = {
        "messages": [dumpd(m) for m in state.get("messages") or []],
        "request_ref": state.get("request_ref"),
        "responses_ref": state.get("responses_ref"),
        "cache_data_ref": state.get("cache_data_ref"),
        "validation_report_ref": state.get("validation_report_ref"),
        "basic_stats_ref": state.get("basic_stats_ref"),
    }
    artifacts: Dict[str, Any] = {}
    if state.get("request_raw") is not None:
        artifacts["request_ref"] = state["request_raw"]
    if state.get("responses_raw") is not None:
        artifacts["responses_ref"] = state["responses_raw"]
    return persisted, artifacts


def hydrate_state(
    persisted: PersistedState,
    request_raw: Optional[Dict[str, Any]] = None,
    responses_raw: Optional[List[Dict[str, Any]]] = None,
) -> GraphState:
    """Reconstruct a live GraphState from its persisted form."""
    messages = [load(m, allowed_objects="messages") for m in persisted.get("messages") or []]

    request = TOPTRequest.from_dict(request_raw) if request_raw else None
    responses = [TOPTResponse.from_dict(r) for r in (responses_raw or [])]

    return {
        "messages": messages,
        "request": request,
        "responses": responses,
        "validation_report": persisted.get("validation_report"),
        "basic_stats": persisted.get("basic_stats"),
        "cache_data": persisted.get("cache_data"),
        "request_ref": persisted.get("request_ref"),
        "responses_ref": persisted.get("responses_ref"),
        "cache_data_ref": persisted.get("cache_data_ref"),
        "validation_report_ref": persisted.get("validation_report_ref"),
        "basic_stats_ref": persisted.get("basic_stats_ref"),
        "request_raw": request_raw,
        "responses_raw": responses_raw,
    }
