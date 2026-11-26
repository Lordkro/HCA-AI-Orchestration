//! Timestamp encoding.
//!
//! Stored as RFC 3339 UTC with microseconds and a `Z` suffix. Every value
//! has the same length and layout, so text comparison in SQL matches time
//! order.

use chrono::{DateTime, SecondsFormat, Utc};

use crate::{Result, StoreError};

pub(crate) fn encode(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Micros, true)
}

pub(crate) fn decode(text: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text)
        .map(|t| t.with_timezone(&Utc))
        .map_err(|e| StoreError::Corrupt(format!("bad timestamp {text:?}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_order_matches_time_order() {
        let a = DateTime::from_timestamp(1_700_000_000, 5_000).unwrap();
        let b = DateTime::from_timestamp(1_700_000_000, 50_000).unwrap();
        let c = DateTime::from_timestamp(1_800_000_000, 0).unwrap();
        assert!(encode(a) < encode(b));
        assert!(encode(b) < encode(c));
        assert_eq!(decode(&encode(b)).unwrap(), b);
    }
}
