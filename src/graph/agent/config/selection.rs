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
