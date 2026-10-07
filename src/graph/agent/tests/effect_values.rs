use super::*;

/// Builds a checkpoint with nontrivial opaque input and controller state.
fn checkpoint() -> Result<EdgeAgentCheckpoint, GraphError> {
    Ok(EdgeAgentCheckpoint {
        version: CHECKPOINT_VERSION,
        phase: EdgeAgentPhase::BeforeModel,
        session_id: "session".into(),
        input: Value::array(["large".repeat(1000).into()]),
        resolved: encode(ResolvedAgentConfig {
            model: "openai:///test".into(),
            instructions: "test".into(),
            memory: Some("memory".into()),
            provider_config: None,
            max_output_tokens: Some(64),
            tools: Vec::new(),
            resources: Vec::new(),
        })?,
        selected_tools: Vec::new(),
        budget: None,
        guidance: None,
        metrics: AgentLoopMetrics::default(),
        control_state: Some(Value::array([true.into()])),
    })
}

/// Direct checkpoint conversion preserves the existing shape and shares opaque user values.
#[test]
fn checkpoint_shape_and_sharing() -> Result<(), GraphError> {
    let original = checkpoint()?;
    let expected = encode(&original)?;
    let value = original.clone().into_value()?;
    assert_eq!(value, expected);
    let decoded = EdgeAgentCheckpoint::from_value(&value)?;
    assert_eq!(encode(&decoded)?, expected);
    assert!(std::ptr::eq(
        original.input.as_array().ok_or_else(invalid)?,
        decoded.input.as_array().ok_or_else(invalid)?
    ));
    Ok(())
}

/// All phase envelopes keep their Serde format, including acknowledged transitions.
#[test]
fn effect_envelope_shapes() -> Result<(), GraphError> {
    let checkpoint = checkpoint()?.into_value()?;
    for effect in [
        AgentEffectCheckpoint::Configure {
            version: CHECKPOINT_VERSION,
            input: checkpoint.clone(),
        },
        AgentEffectCheckpoint::Control {
            version: CHECKPOINT_VERSION,
            checkpoint: checkpoint.clone(),
        },
        AgentEffectCheckpoint::Flush {
            version: CHECKPOINT_VERSION,
            transition: checkpoint.clone(),
        },
        AgentEffectCheckpoint::Generate {
            version: CHECKPOINT_VERSION,
            checkpoint: checkpoint.clone(),
        },
    ] {
        let expected = encode(&effect)?;
        let value = effect.into_value()?;
        assert_eq!(value, expected);
        assert_eq!(
            encode(AgentEffectCheckpoint::from_value(&value)?)?,
            expected
        );
    }
    Ok(())
}

/// Unknown effect fields remain errors instead of being ignored by the borrowed reader.
#[test]
fn unknown_effect_fields_are_rejected() -> Result<(), GraphError> {
    let value = object([
        ("effect", "configure".into()),
        ("version", CHECKPOINT_VERSION.into()),
        ("input", Value::NULL),
        ("surprise", true.into()),
    ])?;
    assert!(AgentEffectCheckpoint::from_value(&value).is_err());
    Ok(())
}
