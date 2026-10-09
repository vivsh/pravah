# Pravah examples

Each example covers a distinct capability. Start with the Chat builder or typed
workflow; the remaining programs focus on one integration or execution boundary.

## Run without credentials

```sh
cargo run --example graph_chat_builder --features testing
cargo run --example graph_typed
```

| Example | Demonstrates | Extra flag |
| --- | --- | --- |
| [graph_chat_builder](graph_chat_builder.rs) | Typed Chat, request memory, message keys, application state and JSON restoration | `--features testing` |
| [graph_chat_stream](graph_chat_stream.rs) | Streaming text previews and authoritative typed completion with a local provider | — |
| [graph_chat_working_memory](graph_chat_working_memory.rs) | Compaction, previous-summary access and completed-history replacement | `--features testing` |
| [graph_json_tools](graph_json_tools.rs) | Catalogue-defined JSON tools, captured operation bindings and a shared Context service | `--features testing` |
| [graph_typed](graph_typed.rs) | Maps, branches, reused subflows and `each` | — |
| [graph_snapshot_resume](graph_snapshot_resume.rs) | Explicit suspension, snapshot and typed resumption | — |
| [graph_external_task](graph_external_task.rs) | External task suspension, snapshot restoration and explicit result delivery | — |
| [graph_agent_budgets](graph_agent_budgets.rs) | Declarative agent construction, agent-turn and per-tool budgets with a deterministic provider | `--features testing` |
| [graph_agent_control](graph_agent_control.rs) | Controller-requested approval, explicit resume and forced conclusion | `--features testing` |
| [graph_diagram](graph_diagram.rs) | Mermaid and DOT output for a split/join workflow | — |
| [graph_untyped](graph_untyped.rs) | Direct graph construction and handler registration | — |
| [graph_json_invocation](graph_json_invocation.rs) | Stateless JSON invocation of a trusted graph | — |

## External service required

| Example | Demonstrates | Requirements |
| --- | --- | --- |
| [graph_agent_mcp](graph_agent_mcp.rs) | MCP text resources and invocation-time tool selection | `--features mcp`, Streamable HTTP MCP server and model credentials |

For example:

```sh
PRAVAH_MCP_URL=https://mcp.example.com \
PRAVAH_MCP_RESOURCE_URI=policy://approvals \
cargo run --example graph_agent_mcp --features mcp -- "How should this be approved?"
```

Set `PRAVAH_MCP_BEARER_TOKEN` when required by the server. Set
`PRAVAH_ALLOW_SEARCH=1` to expose the optional search tool.

Legacy APIs are compatibility-only; their usage is documented in the
[legacy guide](../docs/legacy.md), not duplicated here.
