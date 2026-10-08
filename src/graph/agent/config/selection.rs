use super::McpResourceRef;

/// Validates references and the simple named-template vocabulary supported by Pravah MCP.
pub(crate) fn validate_resources(refs: &[McpResourceRef]) -> Result<(), String> {
    for (index, resource) in refs.iter().enumerate() {
        if refs[..index].contains(resource) {
            return Err("duplicate MCP resource reference".into());
        }
        if resource.server().trim().is_empty() || resource.uri().trim().is_empty() {
            return Err("MCP server and URI must not be empty".into());
        }
        let mut uri = resource.uri().to_owned();
        for name in resource.arguments().keys() {
            let pattern = format!("{{{name}}}");
            if name.is_empty() || !uri.contains(&pattern) {
                return Err("unused or empty MCP template argument".into());
            }
            uri = uri.replace(&pattern, "argument");
        }
        if uri.contains(['{', '}']) || uri.chars().any(char::is_whitespace) || !uri.contains(':') {
            return Err("invalid or unresolved MCP resource URI".into());
        }
    }
    Ok(())
}

/// Checks explicit selection without retaining a registry-derived index.
pub(crate) fn validate_tool_names<'a>(
    names: &[String],
    candidates: impl IntoIterator<Item = &'a str>,
) -> Result<(), String> {
    let candidates: Vec<_> = candidates.into_iter().collect();
    for (index, name) in names.iter().enumerate() {
        if names[..index].contains(name) {
            return Err(format!("duplicate tool '{name}'"));
        }
        if !candidates.contains(&name.as_str()) {
            return Err(format!("unknown tool '{name}'"));
        }
    }
    Ok(())
}
