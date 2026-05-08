use schemars::JsonSchema;
use scraper::node::Node;
use scraper::{ElementRef, Html, Selector};
use serde::{Deserialize, Serialize};

use super::base::{Tool, ToolError};
use crate::context::Context;

const SKIP_TAGS: &[&str] = &["script", "style", "nav", "header", "footer", "aside"];

#[derive(Debug, Serialize)]
pub struct FetchUrlOutput {
    pub url: String,
    pub status: u16,
    pub body: String,
}

#[derive(Debug, Serialize)]
pub struct ScrapeUrlOutput {
    pub url: String,
    pub status: u16,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byline: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub excerpt: Option<String>,
}

/// Rejects any URL whose scheme is not `http` or `https`.
fn check_url_scheme(url: &str) -> Result<(), ToolError> {
    let scheme = url.split(':').next().unwrap_or("");
    if scheme == "http" || scheme == "https" {
        Ok(())
    } else {
        Err(ToolError::Http(format!(
            "only http/https URLs are permitted; got scheme '{scheme}'"
        )))
    }
}

/// Fetches a URL and returns the raw response body as text.
#[derive(Deserialize, JsonSchema)]
pub struct FetchUrl {
    /// The URL to fetch.
    pub url: String,
}

impl Tool for FetchUrl {
    type Output = FetchUrlOutput;
    fn name() -> &'static str {
        "fetch_url"
    }
    fn description() -> &'static str {
        "Fetch a URL and return the raw response body as text."
    }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        check_url_scheme(&self.url)?;
        let resp = ctx
            .http_client()
            .get(&self.url)
            .send()
            .await
            .map_err(|e| ToolError::Http(e.to_string()))?;
        let status = resp.status().as_u16();
        let body = resp
            .text()
            .await
            .map_err(|e| ToolError::Http(e.to_string()))?;
        Ok(FetchUrlOutput {
            url: self.url,
            status,
            body,
        })
    }
}

/// Fetches a URL and returns the main readable content using the Readability algorithm.
#[derive(Deserialize, JsonSchema)]
pub struct ScrapeUrl {
    /// The URL to scrape.
    pub url: String,
}

impl Tool for ScrapeUrl {
    type Output = ScrapeUrlOutput;
    fn name() -> &'static str {
        "scrape_url"
    }
    fn description() -> &'static str {
        "Fetch a URL and extract the main readable content (title, author, text) using Mozilla Readability."
    }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        check_url_scheme(&self.url)?;
        let resp = ctx
            .http_client()
            .get(&self.url)
            .send()
            .await
            .map_err(|e| ToolError::Http(e.to_string()))?;
        let status = resp.status().as_u16();
        let html = resp
            .text()
            .await
            .map_err(|e| ToolError::Http(e.to_string()))?;
        let url = self.url;

        let html_owned = html.clone();
        let url_owned = url.clone();
        let readability_result = tokio::task::spawn_blocking(move || {
            dom_smoothie::Readability::new(html_owned, Some(url_owned.as_str()), None)
                .and_then(|mut r| r.parse())
                .map(|a| (a.title, a.text_content.to_string(), a.byline, a.excerpt))
        })
        .await
        .map_err(|e| ToolError::Http(e.to_string()))?;

        match readability_result {
            Ok((title, text, byline, excerpt)) => Ok(ScrapeUrlOutput {
                url,
                status,
                text,
                title: Some(title),
                byline,
                excerpt,
            }),
            Err(_) => {
                let text = extract_text(&html);
                Ok(ScrapeUrlOutput {
                    url,
                    status,
                    text,
                    title: None,
                    byline: None,
                    excerpt: None,
                })
            }
        }
    }
}

/// Extracts readable text from HTML, preferring `<article>` > `<main>` > `<body>`.
/// Skips `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<aside>` subtrees.
pub fn extract_text(html: &str) -> String {
    let document = Html::parse_document(html);
    for container in &["article", "main", "body"] {
        if let Ok(sel) = Selector::parse(container) {
            if let Some(root) = document.select(&sel).next() {
                let mut buf = Vec::new();
                collect_text(root, &mut buf);
                let text = buf.join(" ");
                if !text.trim().is_empty() {
                    return text;
                }
            }
        }
    }
    String::new()
}

/// Recursively walks an element's subtree, collecting text nodes and skipping boilerplate tags.
fn collect_text(el: ElementRef<'_>, buf: &mut Vec<String>) {
    for child in el.children() {
        match child.value() {
            Node::Text(t) => {
                let s = t.trim();
                if !s.is_empty() {
                    buf.push(s.to_owned());
                }
            }
            Node::Element(e) => {
                if !SKIP_TAGS.contains(&e.name()) {
                    if let Some(child_el) = ElementRef::wrap(child) {
                        collect_text(child_el, buf);
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::FlowConf;

    fn ctx() -> Context {
        Context::new(FlowConf::default())
    }

    /// `FetchUrl` returns the response body and status code from a mock server.
    #[tokio::test]
    async fn fetch_url_returns_body_and_status() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/page")
            .with_status(200)
            .with_body("hello world")
            .create_async()
            .await;
        let url = format!("{}/page", server.url());

        let out = FetchUrl { url: url.clone() }.call(ctx()).await.unwrap();
        assert_eq!(out.status, 200);
        assert_eq!(out.body, "hello world");
        assert_eq!(out.url, url);
    }

    /// `ScrapeUrl` extracts article text and excludes nav/script content.
    /// Uses HTML with enough content density for dom_smoothie's scoring algorithm.
    #[tokio::test]
    async fn scrape_url_extracts_article_text() {
        let html = r#"<!DOCTYPE html><html lang="en">
<head><title>Test Article</title></head>
<body>
<nav id="site-nav">Skip me navigation links home about contact</nav>
<main><article>
  <h1>Keep this content</h1>
  <p>This article has enough text for the readability algorithm to extract it properly and score it above the threshold needed.</p>
  <p>Second paragraph adds more substance so the scoring works correctly across multiple sentences and words.</p>
  <script>ignore js code should not appear in output</script>
</article></main>
<footer>Footer boilerplate to skip</footer>
</body></html>"#;
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/")
            .with_status(200)
            .with_header("content-type", "text/html")
            .with_body(html)
            .create_async()
            .await;

        let out = ScrapeUrl { url: server.url() }.call(ctx()).await.unwrap();
        let text = &out.text;
        assert!(
            text.contains("Keep this content"),
            "expected article text, got: {text}"
        );
        assert!(!text.contains("Skip me"), "nav text should be excluded");
        assert!(
            !text.contains("ignore js"),
            "script text should be excluded"
        );
    }

    /// `ScrapeUrl` returns the main text content; sparse pages fall back to `extract_text`.
    #[tokio::test]
    async fn scrape_url_returns_main_text() {
        let html = r#"<html><body><main><p>Main content here</p></main></body></html>"#;
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/")
            .with_status(200)
            .with_body(html)
            .create_async()
            .await;

        let out = ScrapeUrl { url: server.url() }.call(ctx()).await.unwrap();
        assert!(out.text.contains("Main content here"));
    }

    /// `FetchUrl` returns `ToolError::Http` for an invalid URL.
    #[tokio::test]
    async fn fetch_url_invalid_url_returns_http_error() {
        let err = FetchUrl {
            url: "not_a_url".into(),
        }
        .call(ctx())
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::Http(_)));
    }

    /// `extract_text` falls back to `<body>` when neither `<article>` nor `<main>` exists.
    #[test]
    fn extract_text_falls_back_to_body() {
        let html = "<html><body><p>Body text</p></body></html>";
        let text = extract_text(html);
        assert!(text.contains("Body text"));
    }
}
