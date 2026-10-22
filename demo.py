# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Walk through the three strategies on synthetic data and print what each one saves.

    python demo.py --offline   # local S3 emulator (moto), no AWS account needed
    python demo.py             # real Amazon S3: set SESSION_STATE_BUCKET and AWS_REGION

The demo measures payload size in characters. It does not call a model, so it shows how much
smaller each payload becomes, not the effect on tokens or latency.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import warnings
from typing import Any, Dict, Iterator, List

from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, HumanMessage

# LangChain marks load() as beta. That is upstream status, not a problem with this code; hide
# the notice so the demo output stays readable.
warnings.filterwarnings("ignore", category=LangChainBetaWarning)


def _synthetic_request(rng: random.Random) -> Dict[str, Any]:
    """An optimizer request: resources, tasks, and the rules that constrain them."""
    return {
        "horizon_days": 7,
        "resources": [
            {"resource_id": f"R-{i:05d}", "skills": rng.sample(["A", "B", "C", "D"], 2),
             "max_hours": rng.choice([40, 48, 60])}
            for i in range(2_000)
        ],
        "tasks": [
            {"task_id": f"T-{i:06d}", "skill": rng.choice("ABCD"), "hours": rng.randint(1, 8),
             "window": [rng.randint(0, 100), rng.randint(101, 168)]}
            for i in range(10_000)
        ],
    }


def _synthetic_responses(rng: random.Random) -> List[Dict[str, Any]]:
    """Optimizer responses: one assignment plan per scenario."""
    return [
        {
            "scenario_id": f"S{s}",
            "assignments": [
                {"task_id": f"T-{i:06d}", "resource_id": f"R-{rng.randrange(2_000):05d}",
                 "start_hour": rng.randint(0, 160)}
                for i in range(10_000)
            ],
        }
        for s in (1, 2)
    ]


def _size(value: Any) -> int:
    return len(json.dumps(value, default=str))


def run() -> None:
    """Run the walkthrough against whatever store the environment configures."""
    # Imported here so the offline mode can set up the S3 emulator first.
    from large_json_context.artifact_store import artifact_store
    from large_json_context.compaction import (
        compact_list_result,
        compact_tool_result,
        get_result_page_tool,
    )
    from large_json_context.state import dehydrate_state, hydrate_state

    # Seeded for repeatable output. This generates sample data, not anything security-sensitive.
    rng = random.Random(42)  # nosec B311
    request_raw = _synthetic_request(rng)
    responses_raw = _synthetic_responses(rng)

    print("Strategy 1: persist lightweight state, offload heavy artifacts")
    state = {
        "messages": [HumanMessage("Which tasks are unassigned?"), AIMessage("Checking now.")],
        "request_raw": request_raw,
        "responses_raw": responses_raw,
    }
    live_size = _size(request_raw) + _size(responses_raw)
    persisted, artifacts = dehydrate_state(state)
    # dehydrate_state returns the payloads to upload. Storing them and recording each
    # reference is the caller's job; this is the step that keeps Memory small.
    for ref_field, payload in artifacts.items():
        persisted[ref_field] = artifact_store.put_json(payload)
    print(f"  heavy payloads in live state : {live_size:>12,} chars")
    print(f"  persisted to AgentCore Memory: {_size(persisted):>12,} chars")

    restored = hydrate_state(
        persisted,
        request_raw=artifact_store.get_json(persisted["request_ref"]),
        responses_raw=artifact_store.get_json(persisted["responses_ref"]),
    )
    print(f"  hydrate_state restored {len(restored['messages'])} messages, "
          f"{len(restored['responses'])} responses, request={restored['request'] is not None}")

    print("\nStrategy 2: page large result lists on demand")
    tool_result = responses_raw[0]["assignments"]
    compact = compact_list_result(tool_result)
    print(f"  {f'full list ({len(tool_result):,} items)':<29}: {_size(tool_result):>12,} chars")
    print(f"  what the model receives      : {len(compact):>12,} chars")
    handle = json.loads(compact)["_result_handle"]
    page = get_result_page_tool(handle, offset=50, limit=50)
    print(f"  get_result_page_tool(offset=50, limit=50) -> {len(page['items'])} items "
          f"of {page['total']:,}")

    print("\nStrategy 3: compact oversized dictionary outputs")
    utilization = {
        "scenario_id": "S1",
        "horizon_days": 7,
        "by_resource": {r["resource_id"]: {"hours": rng.randint(0, 60), "tasks": rng.randint(0, 9)}
                        for r in request_raw["resources"]},
        "overloaded": [r["resource_id"] for r in request_raw["resources"][:400]],
    }
    descriptor = compact_tool_result(utilization)
    print(f"  full dictionary              : {_size(utilization):>12,} chars")
    print(f"  what the model receives      : {len(descriptor):>12,} chars")
    print(f"  descriptor fields            : {sorted(json.loads(descriptor)['fields'])}")


@contextlib.contextmanager
def _offline_s3() -> Iterator[None]:
    """Run against moto's in-memory S3 with throwaway credentials and bucket."""
    from moto import mock_aws  # Development dependency, only needed for this mode.

    # Throwaway values for the emulator only. Never real credentials.
    os.environ.pop("AWS_PROFILE", None)
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"  # nosec B105 - throwaway emulator value
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["SESSION_STATE_BUCKET"] = "demo-artifacts"
    with mock_aws():
        import boto3

        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="demo-artifacts")
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="use a local S3 emulator instead of Amazon S3")
    args = parser.parse_args()
    if args.offline:
        with _offline_s3():
            run()
    else:
        run()


if __name__ == "__main__":
    main()
