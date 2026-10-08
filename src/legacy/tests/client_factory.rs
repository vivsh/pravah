use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::clients::{Client, ClientError, ClientOptions, ModelUrl, ProviderFactory};
use crate::testing::ScriptedFactory;

/// Counts construction separately from the scripted backend's request log.
pub(super) struct CountingFactory {
    pub(super) creations: Arc<AtomicUsize>,
    script: ScriptedFactory,
}

impl CountingFactory {
    /// Creates a fixture with an independent construction counter and shared request log.
    pub(super) fn new(script: ScriptedFactory) -> Self {
        Self {
            creations: Arc::new(AtomicUsize::new(0)),
            script,
        }
    }
}

impl ProviderFactory for CountingFactory {
    async fn llm(&self, url: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        self.creations.fetch_add(1, Ordering::SeqCst);
        self.script.llm(url, options).await
    }
}
