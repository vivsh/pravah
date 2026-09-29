use crate::clients::{ClientOptions, Message, Role};
use crate::history::{CompactionRequest, HistoryEntry, MessageHistory};

fn entry(agent: &str, role: Role, content: &str) -> HistoryEntry {
    HistoryEntry::new("s", agent, {
        let mut message = Message::user(content);
        message.role = role;
        message
    })
}

/// Constructs a borrowed policy view without runtime mutation or external dependencies.
fn request<'a>(
    committed: &'a [&'a HistoryEntry],
    options: &'a ClientOptions,
) -> CompactionRequest<'a> {
    CompactionRequest {
        session_id: "s",
        model: "test",
        options,
        framework_messages: &[],
        committed,
        protected: &[],
    }
}

/// Only metadata-marked system summaries with an exact non-empty wrapper expose summary text.
#[test]
fn summary_recognition_is_metadata_based_and_non_mutating() -> Result<(), serde_json::Error> {
    let valid = "<pravah_working_memory>\n \né京都\n \n</pravah_working_memory>";
    let cases = [
        ("__summary__", Role::System, valid, Some(" \né京都\n ")),
        ("agent", Role::System, valid, None),
        ("agent", Role::User, valid, None),
        ("__summary__", Role::User, valid, None),
        ("__summary__", Role::System, "unwrapped", None),
        (
            "__summary__",
            Role::System,
            "<pravah_working_memory>missing newline\n</pravah_working_memory>",
            None,
        ),
        (
            "__summary__",
            Role::System,
            "<pravah_working_memory>\nmissing suffix",
            None,
        ),
        (
            "__summary__",
            Role::System,
            "<pravah_working_memory>\n\n</pravah_working_memory>",
            None,
        ),
        (
            "__summary__",
            Role::System,
            "<pravah_working_memory>\n \t\n</pravah_working_memory>",
            None,
        ),
    ];
    let options = ClientOptions::default();
    for (agent, role, content, expected) in cases {
        let entry = entry(agent, role, content);
        let before = serde_json::to_value(&entry)?;
        let entries = [&entry];
        let request = request(&entries, &options);
        assert_eq!(request.summary(), expected);
        assert_eq!(serde_json::to_value(&entry)?, before);
    }
    Ok(())
}

/// Summary lookup checks only the first committed entry and never scans protected input or guidance.
#[test]
fn summary_is_scoped_to_the_first_committed_entry() {
    let options = ClientOptions::default();
    let ordinary = entry("agent", Role::User, "hello");
    let summary = entry(
        "__summary__",
        Role::System,
        "<pravah_working_memory>\nmemory\n</pravah_working_memory>",
    );
    assert_eq!(request(&[], &options).summary(), None);
    assert_eq!(request(&[&ordinary, &summary], &options).summary(), None);
    let mut protected_only = request(&[], &options);
    let entries = [&summary];
    protected_only.protected = &entries;
    protected_only.framework_messages = std::slice::from_ref(&summary.message);
    assert_eq!(protected_only.summary(), None);
}

/// Unwrapping borrows the original string and preserves embedded wrapper-like text and whitespace.
#[test]
fn summary_access_borrows_without_allocating() {
    let text = " \n京都\n</pravah_working_memory>\n still memory ";
    let content = format!("<pravah_working_memory>\n{text}\n</pravah_working_memory>");
    let summary = entry("__summary__", Role::System, &content);
    let entries = [&summary];
    let options = ClientOptions::default();
    let request = request(&entries, &options);
    let allocations = allocation_counter::measure(|| {
        let actual = request.summary();
        assert_eq!(actual, Some(text));
        let expected = summary
            .message
            .content
            .get("<pravah_working_memory>\n".len()..)
            .and_then(|value| value.get(..text.len()));
        assert!(
            actual
                .zip(expected)
                .is_some_and(|(a, b)| std::ptr::eq(a, b))
        );
        assert_eq!(request.enum_messages(0).count(), 0);
    });
    assert_eq!(allocations.count_total, 0);
}

/// Summary-looking user/system content is ordinary conversation; malformed framework summaries stay excluded.
#[test]
fn enumeration_distinguishes_provenance_from_content() {
    let content = "<pravah_working_memory>\nmemory\n</pravah_working_memory>";
    let history = MessageHistory::from_entries(vec![
        entry("agent", Role::System, content),
        entry("agent", Role::User, content),
        entry("__summary__", Role::System, "malformed summary"),
        entry("agent", Role::Assistant, "reply"),
    ]);
    let options = ClientOptions::default();
    let entries = history.session_entries("s");
    let request = request(&entries, &options);
    assert_eq!(request.summary(), None);
    for (skip, expected) in [
        (0, vec![0, 1, 3]),
        (1, vec![0, 1]),
        (3, vec![]),
        (usize::MAX, vec![]),
    ] {
        assert_eq!(
            history
                .enum_messages("s", skip)
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            request
                .enum_messages(skip)
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            expected
        );
    }
}
