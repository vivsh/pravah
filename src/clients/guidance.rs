use super::Provider;

/// Formats framework guidance without changing provider-owned client options.
pub(crate) fn wrap_system_reminder(provider: &Provider, text: &str) -> String {
    match provider {
        Provider::Anthropic | Provider::Gemini => {
            format!("<system-reminder><critical>{text}</critical></system-reminder>")
        }
        _ => text.to_owned(),
    }
}

/// Requests final structured output; Rath alone owns synthetic output-tool handling.
pub(crate) fn conclusion_message(provider: &Provider) -> String {
    match provider {
        Provider::Anthropic | Provider::Gemini => {
            "<system-reminder><critical>TURN LIMIT REACHED</critical>\
             <constraint>This is your final response turn. Do not call any more tools. \
             Provide your best answer now, following the output format already specified.</constraint>\
             </system-reminder>".to_owned()
        }
        _ => "FINAL TURN: do not call any more tools. \
              Provide your best answer now, following the output format already specified.".to_owned(),
    }
}
