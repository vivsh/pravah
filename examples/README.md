# Pravah Examples

Run an example from the repository root with:

```text
cargo run --example <name>
```

## Graph Workflows

These examples are deterministic and require no provider credentials.

| Example | Demonstrates |
| --- | --- |
| `graph_typed_composition` | Reusing the same typed subflow at multiple call sites |
| `graph_snapshot_resume` | Suspending, serializing state, restoring, and resuming |
| `graph_json_invocation` | Driving a trusted graph through JSON requests |
| `graph_typed` | Typed maps, branches, variables, subflows, and `each` |
| `graph_untyped` | Building an `UntypedGraph` and handler registry directly |
| `graph_agent_budgets` | Deterministic agent and per-tool budgets; run with `--features testing` |

## Diagrams And Local Persistence

| Example | Requirements |
| --- | --- |
| `graph_diagram_complex` | Graphviz `dot`; writes generated files under `target/diagrams` |
| `gen_diagrams` | Compatibility-only legacy diagram output; no external service |
| `snapshot` | Compatibility-only legacy snapshot round trip using a temporary file |

## Provider-Backed Usage

These examples can make paid or external model calls. Load credentials through
the provider's environment variable before running them.

| Example | Requirements |
| --- | --- |
| `chat` | Graph-backed typed chat; Gemini credentials, or edit the configured model URL |
| `linear_flow` | Gemini credentials; compatibility-only legacy flow |
| `split_merge` | Gemini credentials; compatibility-only legacy flow |
| `nested_flow` | Gemini credentials; compatibility-only legacy flow |
| `each_node` | Gemini credentials; compatibility-only legacy flow |
| `tool_flow` | Gemini credentials; compatibility-only legacy tool flow |
| `debate` | Gemini credentials and an optional claim argument; compatibility-only legacy flow |
| `image_prompt` | An image path and credentials for `PRAVAH_MODEL_URL`; compatibility-only legacy flow |
| `story` | `GEMINI_API_KEY`, `FAL_KEY`, a story prompt, and network access for generated image downloads |
| `ollama_client` | A reachable Ollama server and the model selected by `OLLAMA_MODEL` |
| `graph_agent_control` | Model credentials; adaptive guidance, tool visibility, conclusion, suspension, and resume |
| `graph_agent_mcp` | `mcp` feature, Streamable HTTP MCP server, selected resource URI, and model credentials |

Run the MCP resource and tool-filter example with:

```text
PRAVAH_MCP_URL=https://mcp.example.com \
PRAVAH_MCP_RESOURCE_URI=policy://approvals \
cargo run --example graph_agent_mcp --features mcp -- "How should this be approved?"
```

Set `PRAVAH_MCP_BEARER_TOKEN` and `PRAVAH_MCP_TENANT` when required by the
server. Set `PRAVAH_ALLOW_SEARCH=1` to expose the optional search tool.

The compatibility-only examples remain available for existing applications.
New workflow features target Pravah's modern typed API and its underlying
`pravah::graph` runtime.
