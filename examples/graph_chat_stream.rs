//! Streams provisional text from a deterministic local model, then accepts the final reply.
//!
//! Run with `cargo run --example graph_chat_stream`. No credentials or network are needed.

mod support;

use pravah::clients::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, LlmBackend, LlmEvent,
    Message, ModelUrl, Provider, ProviderFactory, ProviderRegistry,
};
use pravah::{Chat, Context};
use support::ExampleError;

struct LocalFactory;

struct LocalModel {
    model: ModelUrl,
    options: ClientOptions,
}

impl ProviderFactory for LocalFactory {
    async fn llm(&self, model: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        Ok(Client::from_backend(LocalModel {
            model: model.clone(),
            options,
        }))
    }
}

impl LlmBackend for LocalModel {
    fn model_url(&self) -> &ModelUrl {
        &self.model
    }

    fn options(&self) -> &ClientOptions {
        &self.options
    }

    async fn execute(&self, _: &[Message]) -> Result<ClientResponse, ClientError> {
        Ok(reply())
    }

    /// A real provider owns its transport; this local fixture supplies the same event contract.
    async fn execute_stream<'a>(
        &'a self,
        _: &[Message],
    ) -> Result<rath::llm::LlmStream<'a>, ClientError> {
        let events = [
            LlmEvent::TextDelta {
                text: "Small ".into(),
            },
            LlmEvent::TextDelta {
                text: "steps.".into(),
            },
            LlmEvent::Completed { response: reply() },
        ];
        Ok(Box::pin(futures::stream::iter(events.into_iter().map(Ok))))
    }
}

fn reply() -> ClientResponse {
    ClientResponse::new(
        Provider::External("local".into()),
        ClientOutput::Output(serde_json::json!("Small steps.")),
    )
}

/// Shows progress separately from the typed, authoritative output stored in conversation history.
#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    let providers = ProviderRegistry::new().register("local", LocalFactory)?;
    let mut chat = Chat::builder::<String, String>()
        .model("local:///demo")
        .instructions("Give concise advice.")
        .build(Context::default().with_providers(providers))?;

    let reply = chat
        .send_stream_with_key("How should I begin?", "question-1", |_, event| {
            if let LlmEvent::TextDelta { text } = event {
                print!("{text}");
            }
            std::future::ready(())
        })
        .await?;
    println!("\nFinal typed reply: {}", reply.output);
    println!(
        "Durable messages: {}",
        chat.snapshot()?.history().entries().len()
    );
    Ok(())
}
