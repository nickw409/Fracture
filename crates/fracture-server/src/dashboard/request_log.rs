use super::dto::{RequestRecord, RequestsResponse};
use std::collections::VecDeque;
use std::sync::Mutex;

const MAX_RECORDS: usize = 1000;

/// Bounded log of completed requests, newest first.
pub struct RequestLog {
    inner: Mutex<VecDeque<RequestRecord>>,
}

impl RequestLog {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(MAX_RECORDS)),
        }
    }

    /// Record a completed request.
    pub fn push(&self, record: RequestRecord) {
        let mut log = self.inner.lock().unwrap();
        if log.len() >= MAX_RECORDS {
            log.pop_back();
        }
        log.push_front(record);
    }

    /// Get a page of records (1-indexed) with the total count.
    pub fn page(&self, page: usize, per_page: usize) -> RequestsResponse {
        let log = self.inner.lock().unwrap();
        let total = log.len();
        let page = page.max(1);
        let per_page = per_page.clamp(1, 100);
        let start = (page - 1) * per_page;

        let requests: Vec<RequestRecord> = log
            .iter()
            .skip(start)
            .take(per_page)
            .cloned()
            .collect();

        RequestsResponse {
            requests,
            total,
            page,
            per_page,
        }
    }
}

impl Default for RequestLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: &str) -> RequestRecord {
        RequestRecord {
            id: id.to_string(),
            request_type: "chat",
            status: "completed",
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            time_to_first_token_ms: 100.0,
            total_duration_ms: 500.0,
            tokens_per_second: 40.0,
            finish_reason: "stop",
            temperature: 0.7,
            created_at: String::new(),
        }
    }

    #[test]
    fn test_empty_log() {
        let log = RequestLog::new();
        let resp = log.page(1, 10);
        assert_eq!(resp.total, 0);
        assert!(resp.requests.is_empty());
        assert_eq!(resp.page, 1);
        assert_eq!(resp.per_page, 10);
    }

    #[test]
    fn test_push_and_page() {
        let log = RequestLog::new();
        log.push(make_record("a"));
        log.push(make_record("b"));
        log.push(make_record("c"));

        let resp = log.page(1, 10);
        assert_eq!(resp.total, 3);
        assert_eq!(resp.requests.len(), 3);
        // Newest first.
        assert_eq!(resp.requests[0].id, "c");
        assert_eq!(resp.requests[1].id, "b");
        assert_eq!(resp.requests[2].id, "a");
    }

    #[test]
    fn test_pagination() {
        let log = RequestLog::new();
        for i in 0..5 {
            log.push(make_record(&format!("r{i}")));
        }

        // Page 1, 2 per page.
        let p1 = log.page(1, 2);
        assert_eq!(p1.requests.len(), 2);
        assert_eq!(p1.total, 5);
        assert_eq!(p1.requests[0].id, "r4");
        assert_eq!(p1.requests[1].id, "r3");

        // Page 2.
        let p2 = log.page(2, 2);
        assert_eq!(p2.requests.len(), 2);
        assert_eq!(p2.requests[0].id, "r2");

        // Page 3 (partial).
        let p3 = log.page(3, 2);
        assert_eq!(p3.requests.len(), 1);
        assert_eq!(p3.requests[0].id, "r0");

        // Page 4 (empty).
        let p4 = log.page(4, 2);
        assert!(p4.requests.is_empty());
    }

    #[test]
    fn test_bounded_capacity() {
        let log = RequestLog::new();
        for i in 0..1005 {
            log.push(make_record(&format!("r{i}")));
        }
        let resp = log.page(1, 100);
        assert_eq!(resp.total, 1000);
        // Oldest entries evicted.
        assert_eq!(resp.requests[0].id, "r1004");
    }

    #[test]
    fn test_page_zero_treated_as_one() {
        let log = RequestLog::new();
        log.push(make_record("a"));
        let resp = log.page(0, 10);
        assert_eq!(resp.page, 1);
        assert_eq!(resp.requests.len(), 1);
    }

    #[test]
    fn test_per_page_clamped() {
        let log = RequestLog::new();
        for i in 0..200 {
            log.push(make_record(&format!("r{i}")));
        }
        // per_page > 100 clamped to 100.
        let resp = log.page(1, 500);
        assert_eq!(resp.per_page, 100);
        assert_eq!(resp.requests.len(), 100);
    }
}
