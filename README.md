# Reducing token costs and latency in Amazon Bedrock AgentCore JSON workloads

Companion code for the AWS Machine Learning Blog post *Reducing token costs and latency in Amazon
Bedrock AgentCore JSON workloads*. <!-- TODO: add the post URL once it is published. -->

When an agent's tools return large JSON documents, each call can push hundreds of thousands of
characters into the model's context. This repository contains the code from the post for three
strategies that keep those payloads out of the context while leaving the full data available:

1. **Persist lightweight session state, offload heavy artifacts.** Store large payloads in Amazon
   Simple Storage Service (Amazon S3), and keep only references in Amazon Bedrock AgentCore Memory.
2. **Page large result lists on demand.** Return a count, a sample, and a handle. The model fetches
   more pages only when it needs them.
3. **Compact oversized dictionary outputs.** Return a description of the dictionary's structure plus
   a handle, instead of the dictionary itself.

## What this repository does and doesn't do

The code is a set of building blocks you add to your own agent. It isn't a complete agent and
doesn't call a model.

The included demo shows how much smaller each payload becomes. Token counts and latency depend on
your model, your prompts, and how often the model fetches more detail, so the demo can't reproduce
the figures in the post. Measure them in your own workload, for example with Amazon Bedrock
AgentCore Observability.

## Repository structure

```
large_json_context/
├── artifact_store.py   # Strategy 1: content-addressed S3 store
├── state.py            # Strategy 1: GraphState, PersistedState, dehydrate_state, hydrate_state
└── compaction.py       # Strategies 2 and 3: list paging and dictionary descriptors
demo.py                 # Walkthrough on synthetic data
tests/                  # Unit tests, run against an in-memory S3 emulator
```

## Prerequisites

- Python 3.12 or later.
- To run the demo offline: nothing else. It uses [moto](https://github.com/getmoto/moto), an
  in-memory S3 emulator.
- To use real Amazon S3: an AWS account, an S3 bucket, and credentials that allow `s3:GetObject` and
  `s3:PutObject` on that bucket's artifact prefix.

## Getting started

```bash
git clone https://github.com/aws-samples/sample-reducing-token-costs-and-latency-in-amazon-bedrock-agentcore-json-workloads.git
cd sample-reducing-token-costs-and-latency-in-amazon-bedrock-agentcore-json-workloads
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run the tests:

```bash
pytest
```

Run the demo without an AWS account:

```bash
python demo.py --offline
```

Expected output:

```
Strategy 1: persist lightweight state, offload heavy artifacts
  heavy payloads in live state :    2,239,566 chars
  persisted to AgentCore Memory:          614 chars
  hydrate_state restored 2 messages, 2 responses, request=True

Strategy 2: page large result lists on demand
  full list (10,000 items)     :      693,161 chars
  what the model receives      :          481 chars
  get_result_page_tool(offset=50, limit=50) -> 50 items of 10,000

Strategy 3: compact oversized dictionary outputs
  full dictionary              :       80,122 chars
  what the model receives      :          376 chars
  descriptor fields            : ['by_resource', 'horizon_days', 'overloaded', 'scenario_id']
```

To run the demo against your own bucket instead:

```bash
export SESSION_STATE_BUCKET=<your-bucket-name>
export AWS_REGION=<your-bucket-region>
python demo.py
```

The demo writes objects under the `artifacts/` prefix of that bucket. See [Clean up](#clean-up).

## Using the code in your agent

1. Set `SESSION_STATE_BUCKET` and `AWS_REGION`. See [Configuration](#configuration).
2. At the end of each turn, call `dehydrate_state(state)`. Upload each returned artifact with
   `artifact_store.put_json(payload)`, store the returned reference in the matching `*_ref` field,
   and write the persisted state to AgentCore Memory.
3. When a session resumes, read the persisted state from AgentCore Memory and call `hydrate_state`.
   Fetch the heavy payloads with `artifact_store.get_json(ref)` only when a tool needs them.
4. Pass list results through `compact_list_result`, and dictionary results through
   `compact_tool_result`, before returning them to the model.
5. Register `get_result_page_tool` as a tool the model can call to read more of a stored result.

## About `TOPTRequest` and `TOPTResponse`

TOPT stands for Training Optimizer, the optimization service whose requests and responses the
agent in the post works with. The post uses two domain classes, `TOPTRequest` and `TOPTResponse`,
without defining them. They hold the optimizer's request JSON and response JSON after parsing them
into typed objects that tools can work with. Their fields depend on your optimizer, so the post
doesn't show them.

`large_json_context/state.py` includes minimal stand-ins that keep the raw dictionary in a `raw`
attribute, so the code runs as-is. Replace them with your own classes, and keep a `from_dict`
class method on each, because `hydrate_state` calls it.

## Reference

Every name used in the code, in the order the post introduces it.

### State and persistence (`state.py`)

| Name | What it is |
|---|---|
| `OriginalGraphState` | The agent state before Strategy 1, with every heavy payload held inline. The post shows it as its first `GraphState`; it's renamed here so both versions can exist together. |
| `GraphState` | The live LangGraph state: conversation messages, the parsed domain objects, and a reference for each heavy payload. |
| `PersistedState` | What's written to AgentCore Memory: the messages and the artifact references only. |
| `TOPTRequest`, `TOPTResponse` | Typed versions of the Training Optimizer (TOPT) request and response JSON. See [About `TOPTRequest` and `TOPTResponse`](#about-toptrequest-and-toptresponse). |
| `messages` | The conversation. In `GraphState`, the `add_messages` reducer appends new messages instead of replacing the list. |
| `request`, `responses` | The parsed optimizer request, and the parsed responses (one per scenario). |
| `validation_report` | The output of validating the optimizer's response against business rules. |
| `basic_stats` | Summary statistics the tools compute for quick answers. |
| `cache_data` | Results the tools reuse between queries. |
| `request_raw`, `responses_raw` | The raw JSON for the request and responses, when loaded in the current session. |
| `*_ref` fields | The artifact reference for the matching payload: `request_ref`, `responses_ref`, `cache_data_ref`, `validation_report_ref`, `basic_stats_ref`. |
| `dehydrate_state` | Converts live state into `PersistedState`, and returns the heavy payloads you need to upload. |
| `hydrate_state` | Rebuilds live state from `PersistedState` and any raw payloads you pass in. |
| `dumpd`, `load` | LangChain functions that serialize messages to plain dictionaries and restore them. `allowed_objects="messages"` restricts `load` to message types, which matters because persisted state is read back from storage. |

### Artifact store (`artifact_store.py`)

| Name | What it is |
|---|---|
| `S3ArtifactStore` | Stores JSON payloads in Amazon S3 under a key derived from their content. |
| `_canonical_bytes` | Serializes a payload with sorted keys and compact separators, so identical content always produces identical bytes. |
| `_content_hash` | The SHA-256 digest of those bytes, as 64 lowercase hex characters. It's used as the artifact reference. |
| `put_json` | Stores a payload and returns its reference. Skips the upload if this process has already stored the same content. |
| `get_json` | Returns the payload for a reference, from the in-process cache if available, otherwise from Amazon S3. |
| `_key` | Builds the S3 object key, `<prefix>/<reference>.json`. |
| `artifact_store` | The shared store instance the other modules use. It's created on first use. |
| `get_artifact_store`, `set_artifact_store` | Return or replace the shared instance. `set_artifact_store` is mainly useful in tests. |
| `logger` | The module logger. It logs the bucket name and prefix when the store is created. |

### Compaction (`compaction.py`)

| Name | What it is |
|---|---|
| `_LLM_SAMPLE_SIZE` | Lists with more items than this (default `5`) are stored and returned as a sample. |
| `_MAX_TOOL_CONTENT_CHARS` | Serialized dictionary results longer than this (default `7600`, about 4,000 tokens of dense JSON) are replaced by a descriptor. |
| `compact_list_result` | Strategy 2. Returns a short list in full, or stores a long one and returns `total`, `sample`, and `_result_handle`. |
| `get_result_page_tool` | Strategy 2. A tool the model calls with a handle, `offset`, and `limit` to read a slice of a stored list. |
| `_oversized_descriptor` | Strategy 3. Describes a dictionary: each list field as its count and a sample, each nested dictionary as its key count, and each scalar as its value. Fields starting with `_` are skipped. |
| `compact_tool_result` | Strategy 3. Returns a dictionary in full, or stores it and returns its descriptor if it's longer than `_MAX_TOOL_CONTENT_CHARS`. |
| `llm_view`, `full_view` | Arguments to `compact_tool_result`: the version of the result the model would see, and the complete result to store. They're usually the same. |
| `_oversized`, `_result_handle` | Keys in a compacted result. `_oversized` tells the model it's looking at a summary; `_result_handle` is the reference to pass to `get_result_page_tool`. |

## Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `SESSION_STATE_BUCKET` | Yes | none | S3 bucket for artifacts. |
| `ARTIFACT_STORE_PREFIX` | No | `artifacts` | Key prefix inside the bucket. |
| `AWS_REGION` | See note | none | Region of the bucket. |
| `AWS_PROFILE` | No | none | Named AWS profile to use for credentials. |

**Region:** the store uses, in order, the `region` argument, `AWS_REGION`, and then the standard
boto3 resolution: `AWS_DEFAULT_REGION`, the region in your AWS profile, and instance metadata. If
none of these is set, the store raises an error. Without one, the S3 client falls back to
`us-east-1` without reporting it, which writes artifacts to an unexpected Region.

## Differences from the code in the post

The post's code is reproduced as closely as possible. Each change is marked in the source with a
comment that starts with `Differs from the post`.

| Change | Why |
|---|---|
| No default Region; the store raises an error if none is configured | The post defaults to a specific Region. See the Region note above. |
| References are validated before they're used in an S3 key | The model passes references to `get_result_page_tool`. Accepting only the 64-character digest format prevents a crafted value from addressing another object. |
| The store is created on first use, not at import | The post creates it at import time, which raises an error whenever `SESSION_STATE_BUCKET` isn't set, including in tests. |
| The first `GraphState` is renamed `OriginalGraphState`, and its docstring is indented | Both versions need to exist together. In the post, the docstring starts at column 0, which is a Python `IndentationError`. |
| `request_raw` and `responses_raw` are declared in `GraphState` | The functions read and write them, but the post's `GraphState` doesn't list them. |
| The paging snippet and the size check are wrapped in `compact_list_result` and `compact_tool_result` | In the post, they're fragments that use `return` outside a function. |
| `_canonical_bytes`, `_LLM_SAMPLE_SIZE`, `_MAX_TOOL_CONTENT_CHARS`, `logger`, and minimal `TOPTRequest` / `TOPTResponse` are defined | The post uses them without defining them. |

## Limitations

This is sample code. Before using it in production, consider these limitations:

- `get_result_page_tool` doesn't use its `field` argument, returns `None` for stored dictionaries,
  and doesn't cap `limit`. Add those if your model needs them.
- `_oversized_descriptor` samples the first five items of each list, however large each item is. A
  list holding a few very large items can produce a descriptor larger than expected.
- The in-process cache in `S3ArtifactStore` has no size limit. In a long-running process, bound it or
  clear it between sessions.
- References aren't tied to a session. Any code holding a reference can read the artifact. If
  several users' sessions share a bucket, add the session ID to the key, or check ownership before
  reading.

## Security

The artifact store sends data only to the bucket you configure. Keep the bucket private: leave
S3 Block Public Access on, use server-side encryption, and grant only `s3:GetObject` and
`s3:PutObject` on the artifact prefix. Artifacts can contain your business data, so apply the same
access controls and retention policy you'd apply to the source documents.

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for how to report a security
issue.

## Costs

The demo's offline mode uses no AWS resources. When you use Amazon S3, you pay for storage and
requests; see [Amazon S3 pricing](https://aws.amazon.com/s3/pricing/). In your own agent, you also
pay for Amazon Bedrock model usage and the AgentCore features you use; see
[Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/).

## Clean up

If you ran the demo against your own bucket, delete what it wrote:

```bash
aws s3 rm "s3://$SESSION_STATE_BUCKET/artifacts/" --recursive
```

This deletes every object under that prefix. If your bucket has versioning enabled, earlier
versions remain until you delete them or a lifecycle rule expires them.

## Disclaimer

This is sample code, for non-production usage. You should work with your security and legal teams
to meet your organizational security, regulatory and compliance requirements before deployment.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
