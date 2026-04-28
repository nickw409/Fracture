//! Simple config file loader.
//!
//! Loads KEY=VALUE pairs from a file into a `FractureConfig` map.
//! Binaries use `config.get("KEY")` with CLI flag fallback.
//!
//! File format:
//! - One KEY=VALUE per line
//! - Lines starting with `#` are comments
//! - Empty lines are ignored
//! - Values can be optionally quoted with double quotes

use std::collections::HashMap;
use std::path::Path;

/// Loaded config values from a fracture.env file.
#[derive(Debug, Default)]
pub struct FractureConfig {
    values: HashMap<String, String>,
}

impl FractureConfig {
    /// Get a config value by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    /// Get a config value, falling back to a CLI flag value.
    /// Checks: config file → CLI flag.
    pub fn get_or_flag<'a>(&'a self, key: &str, args: &'a [String], flag: &str) -> Option<&'a str> {
        self.get(key).or_else(|| {
            args.iter()
                .position(|a| a == flag)
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str())
        })
    }

    /// Check if a boolean flag is set (either in config or CLI args).
    pub fn has_flag(&self, key: &str, args: &[String], flag: &str) -> bool {
        self.get(key).is_some_and(|v| v == "true" || v == "1")
            || args.iter().any(|a| a == flag)
    }
}

/// Load a config file.
///
/// Searches in this order:
/// 1. `--config <path>` CLI flag
/// 2. `./fracture.env` (current directory)
/// 3. `/etc/fracture/fracture.env`
///
/// Returns the config and the path it was loaded from.
pub fn load_config(args: &[String]) -> (FractureConfig, Option<String>) {
    let path = args
        .iter()
        .position(|a| a == "--config")
        .and_then(|i| args.get(i + 1).cloned())
        .or_else(|| {
            let p = Path::new("fracture.env");
            p.exists().then(|| p.to_string_lossy().into_owned())
        })
        .or_else(|| {
            let p = Path::new("/etc/fracture/fracture.env");
            p.exists().then(|| p.to_string_lossy().into_owned())
        });

    match path {
        Some(p) => match parse_env_file(&p) {
            Ok(config) => (config, Some(p)),
            Err(_) => (FractureConfig::default(), None),
        },
        None => (FractureConfig::default(), None),
    }
}

/// Parse a config file into a FractureConfig.
fn parse_env_file(path: &str) -> std::io::Result<FractureConfig> {
    let contents = std::fs::read_to_string(path)?;
    let mut values = HashMap::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim().to_string();
            let mut value = value.trim();
            // Strip optional double quotes
            if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
                value = &value[1..value.len() - 1];
            }
            values.insert(key, value.to_string());
        }
    }
    Ok(FractureConfig { values })
}

/// Validate that every `--<name>` token in `args` is present in `known`.
/// Returns the offending flag name (without leading `--`) on the first
/// unknown match.
///
/// Skips `args[0]` (binary path) and the bare `--` separator. Strips
/// `=value` suffixes before matching, so both `--flag value` and
/// `--flag=value` forms work. Tokens that don't start with `--` are
/// treated as positional values and ignored.
pub fn validate_known_flags(args: &[String], known: &[&str]) -> Result<(), String> {
    for arg in args.iter().skip(1) {
        if arg == "--" {
            continue;
        }
        let Some(rest) = arg.strip_prefix("--") else {
            continue;
        };
        let name = rest.split('=').next().unwrap_or(rest);
        if name.is_empty() {
            continue;
        }
        if !known.contains(&name) {
            return Err(name.to_string());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_parse_env_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.env");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# Comment").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "COORDINATOR=192.168.1.10:9400").unwrap();
        writeln!(f, "SEEDS = host1:9400,host2:9400").unwrap();
        writeln!(f, "MODEL=\"/path/to/model.gguf\"").unwrap();

        let config = parse_env_file(path.to_str().unwrap()).unwrap();
        assert_eq!(config.get("COORDINATOR"), Some("192.168.1.10:9400"));
        assert_eq!(config.get("SEEDS"), Some("host1:9400,host2:9400"));
        assert_eq!(config.get("MODEL"), Some("/path/to/model.gguf"));
        assert_eq!(config.get("MISSING"), None);
    }

    #[test]
    fn test_get_or_flag() {
        let config = FractureConfig {
            values: HashMap::from([("KEY".into(), "from_config".into())]),
        };
        let args: Vec<String> = vec!["bin".into(), "--flag".into(), "from_flag".into()];

        // Config takes precedence
        assert_eq!(config.get_or_flag("KEY", &args, "--flag"), Some("from_config"));

        // Falls back to flag when key not in config
        assert_eq!(config.get_or_flag("OTHER", &args, "--flag"), Some("from_flag"));

        // None when neither
        assert_eq!(config.get_or_flag("OTHER", &args, "--missing"), None);
    }

    #[test]
    fn test_has_flag() {
        let config = FractureConfig {
            values: HashMap::from([("BATCHED".into(), "true".into())]),
        };
        let args: Vec<String> = vec!["bin".into(), "--verbose".into()];

        assert!(config.has_flag("BATCHED", &args, "--batched"));
        assert!(config.has_flag("MISSING", &args, "--verbose"));
        assert!(!config.has_flag("MISSING", &args, "--missing"));
    }

    #[test]
    fn test_load_config_with_explicit_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("custom.env");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "PORT=9400").unwrap();

        let args = vec![
            "bin".to_string(),
            "--config".to_string(),
            path.to_str().unwrap().to_string(),
        ];
        let (config, loaded_path) = load_config(&args);
        assert!(loaded_path.is_some());
        assert_eq!(config.get("PORT"), Some("9400"));
    }

    #[test]
    fn test_load_config_no_file() {
        let args = vec!["bin".to_string()];
        let (config, loaded_path) = load_config(&args);
        assert!(loaded_path.is_none());
        assert_eq!(config.get("ANYTHING"), None);
    }

    #[test]
    fn test_validate_known_flags_accepts_known() {
        let args: Vec<String> = ["bin", "--model", "x.gguf", "--port", "8080"]
            .iter().map(|s| s.to_string()).collect();
        assert!(validate_known_flags(&args, &["model", "port"]).is_ok());
    }

    #[test]
    fn test_validate_known_flags_rejects_unknown() {
        let args: Vec<String> = ["bin", "--model", "x.gguf", "--workers", "1"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(
            validate_known_flags(&args, &["model", "min-workers"]),
            Err("workers".to_string())
        );
    }

    #[test]
    fn test_validate_known_flags_handles_equals_form() {
        let args: Vec<String> = ["bin", "--model=x.gguf", "--bogus=value"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(
            validate_known_flags(&args, &["model"]),
            Err("bogus".to_string())
        );
    }

    #[test]
    fn test_validate_known_flags_skips_positional_values() {
        // Values that follow flags must NOT be matched against the known list.
        let args: Vec<String> = ["bin", "--listen", "127.0.0.1:9400"]
            .iter().map(|s| s.to_string()).collect();
        assert!(validate_known_flags(&args, &["listen"]).is_ok());
    }

    #[test]
    fn test_validate_known_flags_ignores_double_dash_separator() {
        let args: Vec<String> = ["bin", "--model", "x.gguf", "--", "extra"]
            .iter().map(|s| s.to_string()).collect();
        assert!(validate_known_flags(&args, &["model"]).is_ok());
    }

    #[test]
    fn test_validate_known_flags_empty_args() {
        let args: Vec<String> = vec!["bin".to_string()];
        assert!(validate_known_flags(&args, &["model"]).is_ok());
    }

    #[test]
    fn test_validate_known_flags_first_unknown_wins() {
        let args: Vec<String> = ["bin", "--first-bad", "--second-bad"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(
            validate_known_flags(&args, &[]),
            Err("first-bad".to_string())
        );
    }
}
