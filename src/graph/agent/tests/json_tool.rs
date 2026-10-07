use super::*;
use serde_json::json;

fn contract(output: JsonValue) -> JsonToolContract {
    JsonToolContract::new(
        ToolDefinition::new(
            "lookup".into(),
            "Look up".into(),
            json!({
                "type":"object", "properties":{"id":{"type":"integer","minimum":1}},
                "required":["id"], "additionalProperties":false,
            }),
        ),
        Some(output),
    )
    .expect("valid contract")
}

/// Native namespacing preserves the contract and changes only the handler identity.
#[test]
fn namespacing_preserves_json_contract() {
    let contract = contract(json!({"type":["string","null"]}));
    let mut payload = Value::object([
        ("tool_handler_key", "old".into()),
        ("json_contract", contract.definition.clone()),
    ])
    .expect("payload");
    namespace_payload_handler(&mut payload, "parent::child");
    assert_eq!(
        payload.get("tool_handler_key").and_then(Value::as_str),
        Some("parent::child")
    );
    assert_eq!(payload.get("json_contract"), Some(&contract.definition));
}

/// Successful strings remain strings rather than being parsed into JSON objects.
#[test]
fn json_string_output_is_exact() {
    let contract = contract(json!({"type":"string"}));
    let json = json!("{\"id\":42}");
    let output = to_value(EdgeToolResult::Success {
        value: json.clone(),
    })
    .expect("envelope");
    let rendered = render_json(&contract, output).expect("render");
    assert_eq!(rendered.message.content, json.to_string());
    assert_eq!(serde_json::to_value(&rendered.value).expect("JSON"), json);
}

/// Error envelopes bypass successful result validation and marker-shaped results stay successful.
#[test]
fn envelopes_preserve_success_and_error_identity() {
    let contract = contract(json!({"type":"object"}));
    let marker = json!({"kind":"error","value":{"error":true}});
    let output = to_value(EdgeToolResult::Success { value: marker }).expect("envelope");
    assert!(!render_json(&contract, output).expect("render").error);
    let error = to_value(EdgeToolResult::Error {
        value: json!("denied"),
    })
    .expect("envelope");
    assert!(render_json(&contract, error).expect("render").error);
}

/// Handler registration cannot validate a different authored input or output contract.
#[test]
fn mismatched_handler_contract_is_rejected() {
    let first = Arc::new(contract(json!({"type":"string"})));
    let second = contract(json!({"type":"integer"}));
    let function = FunctionTool::json(first, |_, _| async { Ok(json!("ok")) });
    let payload = Value::object([("json_contract", second.definition)]).expect("payload");
    assert!(function.validate_payload(&payload).is_err());
}

/// Optional result validation is explicitly unrestricted, while false rejects every success.
#[test]
fn unrestricted_and_false_output_schemas() {
    let definition = ToolDefinition::new("any".into(), "".into(), json!({"type":"object"}));
    let any = JsonToolContract::new(definition, None).expect("contract");
    assert_eq!(
        any.definition.get("output_schema"),
        Some(&Value::from(true))
    );
    for value in [json!(null), json!(7), json!(["x"]), json!("{\"x\":1}")] {
        assert!(any.validate(&value, true).is_ok());
        assert!(contract(json!(false)).validate(&value, true).is_err());
    }
}

/// Contract diagnostics expose locations but never argument or response secrets.
#[test]
fn validation_diagnostics_omit_values() {
    let contract = contract(json!({"type":"integer"}));
    let error = contract
        .validate(&json!("private-response"), true)
        .expect_err("invalid");
    assert!(!error.contains("private-response"));
    assert!(error.contains("schema /type at instance"));
}

/// Authored graphs cannot downgrade JSON definitions or substitute a different parent projection.
#[test]
fn child_bindings_and_version_gate_are_checked() {
    let agent = Agent::<String>::root().tools(|tools| {
        tools.json(
            ToolDefinition::new("lookup".into(), "Look up".into(), json!({"type":"object"})),
            None,
            |_, _| async { Ok(json!(null)) },
        )
    });
    let mut build = build_agent::<String, String>(agent);
    build.payload.agent_id = "agent".into();
    build.payload.configure_handler_key = "agent".into();
    let payload = to_value(&build.payload).expect("agent payload");
    assert!(validate_json_children(&payload, &build.children).is_ok());
    let mut changed = serde_json::to_value(&payload).expect("JSON");
    for (pointer, replacement) in [
        ("/version", json!(PAYLOAD_VERSION)),
        (
            "/tools/0/parameters",
            json!({"type":"object","required":["id"]}),
        ),
        ("/tools/0/child_index", json!(1)),
    ] {
        *changed.pointer_mut(pointer).expect("field") = replacement;
        assert!(validate_json_children(&to_value(&changed).expect("VM"), &build.children).is_err());
        changed = serde_json::to_value(&payload).expect("JSON");
    }
}

/// Typed-only authoring retains payload version six and omits the new child contract field.
#[test]
fn typed_only_definition_keeps_its_format() {
    async fn typed(input: String, _: Context) -> Result<String, ToolError> {
        Ok(input)
    }
    let build = build_agent::<String, String>(Agent::root().tools(|tools| tools.tool(typed)));
    assert_eq!(build.payload.version, PAYLOAD_VERSION);
    for graph in &build.children {
        assert!(child_contract(graph).is_none());
        for node in &graph.nodes {
            if let NodeKind::Continuation { payload, .. } = &node.kind {
                assert!(payload.get("json_contract").is_none());
            }
        }
    }
}
