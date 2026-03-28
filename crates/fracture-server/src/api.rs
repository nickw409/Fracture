use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub text: Option<String>,
    pub message: Option<ResponseMessage>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_response_serialization() {
        let resp = CompletionResponse {
            id: "test-id".to_string(),
            object: "text_completion".to_string(),
            created: 1234567890,
            choices: vec![Choice {
                index: 0,
                text: Some("Hello".to_string()),
                message: None,
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "test-id");
        assert_eq!(json["object"], "text_completion");
        assert_eq!(json["choices"][0]["text"], "Hello");
        assert!(json["choices"][0]["message"].is_null());
        assert_eq!(json["usage"]["prompt_tokens"], 5);
        assert_eq!(json["usage"]["total_tokens"], 6);
    }

    #[test]
    fn test_chat_completion_response_serialization() {
        let resp = CompletionResponse {
            id: "test-id".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            choices: vec![Choice {
                index: 0,
                text: None,
                message: Some(ResponseMessage {
                    role: "assistant".to_string(),
                    content: "Hi there".to_string(),
                }),
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["message"]["content"], "Hi there");
        assert!(json["choices"][0]["text"].is_null());
    }

    #[test]
    fn test_completion_request_defaults() {
        let json = r#"{"prompt": "hello"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "hello");
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.temperature, 1.0);
        assert_eq!(req.top_p, 1.0);
        assert_eq!(req.top_k, 0);
        assert!(!req.stream);
        assert!(req.model.is_none());
    }

    #[test]
    fn test_chat_completion_request_defaults() {
        let json = r#"{"messages": [{"role": "user", "content": "hello"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.temperature, 1.0);
        assert_eq!(req.top_p, 1.0);
        assert!(!req.stream);
    }

    #[test]
    fn test_completion_request_with_overrides() {
        let json = r#"{"prompt": "hi", "max_tokens": 100, "temperature": 0.5, "top_p": 0.9, "top_k": 50, "stream": true, "model": "llama-3-8b"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "hi");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.temperature, 0.5);
        assert_eq!(req.top_p, 0.9);
        assert_eq!(req.top_k, 50);
        assert!(req.stream);
        assert_eq!(req.model.unwrap(), "llama-3-8b");
    }
}
