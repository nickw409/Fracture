use byteorder::{LittleEndian, ReadBytesExt};
use fracture_core::{DType, FractureError, ModelConfig, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

/// GGUF magic number: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x46554747;
/// Only GGUF version 3 is supported.
const GGUF_VERSION: u32 = 3;

/// GGUF metadata value types.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufMetadataType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufMetadataType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(FractureError::GgufParse(format!(
                "unknown metadata type: {v}"
            ))),
        }
    }
}

/// GGUF tensor data types.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum GgufDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4 is unused
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq2Xxs = 15,
    Iq2Xs = 16,
    Iq3Xxs = 17,
    Iq1S = 18,
    Iq4Nl = 19,
    Iq3S = 20,
    Iq2S = 21,
    Iq4Xs = 22,
    I8 = 23,
    I16 = 24,
    I32 = 25,
    I64 = 26,
    F64 = 27,
    Iq1M = 28,
    Bf16 = 30,
}

impl GgufDType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Iq2Xxs),
            16 => Ok(Self::Iq2Xs),
            17 => Ok(Self::Iq3Xxs),
            18 => Ok(Self::Iq1S),
            19 => Ok(Self::Iq4Nl),
            20 => Ok(Self::Iq3S),
            21 => Ok(Self::Iq2S),
            22 => Ok(Self::Iq4Xs),
            23 => Ok(Self::I8),
            24 => Ok(Self::I16),
            25 => Ok(Self::I32),
            26 => Ok(Self::I64),
            27 => Ok(Self::F64),
            28 => Ok(Self::Iq1M),
            30 => Ok(Self::Bf16),
            _ => Err(FractureError::GgufParse(format!(
                "unknown tensor dtype: {v}"
            ))),
        }
    }

    fn to_fracture_dtype(self) -> Result<DType> {
        match self {
            Self::F32 => Ok(DType::FP32),
            Self::F16 => Ok(DType::FP16),
            Self::Bf16 => Ok(DType::BF16),
            Self::I8 => Ok(DType::INT8),
            other => Err(FractureError::UnsupportedDType(format!(
                "GGUF dtype {:?} not yet supported",
                other
            ))),
        }
    }
}

/// A typed metadata value from a GGUF file.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Uint64(v) => Some(*v as u32),
            Self::Int32(v) => Some(*v as u32),
            Self::Uint16(v) => Some(*v as u32),
            Self::Uint8(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Uint32(v) => Some(*v as u64),
            Self::Int64(v) => Some(*v as u64),
            Self::Int32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float64(v) => Some(*v),
            Self::Float32(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Parsed GGUF tensor descriptor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub offset: u64,
    #[allow(dead_code)] // retained for future quantization support
    pub(crate) gguf_dtype: u32,
}

/// Result of parsing a GGUF file.
pub struct GgufFile {
    // Debug is manually implemented below because Mmap doesn't derive Debug.
    pub config: ModelConfig,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub mmap: Mmap,
    pub tensor_data_offset: usize,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("config", &self.config)
            .field("metadata_keys", &self.metadata.keys().collect::<Vec<_>>())
            .field("tensors", &self.tensors.len())
            .field("tensor_data_offset", &self.tensor_data_offset)
            .field("mmap_len", &self.mmap.len())
            .finish()
    }
}

/// Parses GGUF v3 binary files.
pub struct GgufParser;

impl GgufParser {
    /// Parse a GGUF file, returning model config, tensor descriptors, and memory-mapped data.
    pub fn parse(path: &Path) -> Result<GgufFile> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = Cursor::new(mmap.as_ref());

        // Header
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(FractureError::GgufParse(format!(
                "invalid magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"
            )));
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version != GGUF_VERSION {
            return Err(FractureError::GgufParse(format!(
                "unsupported GGUF version: {version} (only v3 supported)"
            )));
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        tracing::info!(
            "GGUF v{version}: {tensor_count} tensors, {metadata_kv_count} metadata entries"
        );

        // Metadata
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut cursor)?;
            let value = read_metadata_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Tensor info table
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut cursor)?;
            let ndims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                shape.push(cursor.read_u64::<LittleEndian>()? as usize);
            }
            // GGUF stores dimensions innermost-first. Reverse to match the
            // row-major convention used by the engine.
            shape.reverse();
            let dtype_code = cursor.read_u32::<LittleEndian>()?;
            let gguf_dtype = GgufDType::from_u32(dtype_code)?;
            let dtype = gguf_dtype.to_fracture_dtype()?;
            let offset = cursor.read_u64::<LittleEndian>()?;
            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                offset,
                gguf_dtype: dtype_code,
            });
        }

        // Tensor data starts after the header, aligned to GGUF_DEFAULT_ALIGNMENT (32 bytes for v3)
        let current_pos = cursor.position() as usize;
        let alignment = get_alignment(&metadata);
        let tensor_data_offset = align_offset(current_pos, alignment);

        tracing::info!(
            "tensor data starts at offset 0x{tensor_data_offset:X} (alignment={alignment})"
        );

        let config = extract_model_config(&metadata)?;

        Ok(GgufFile {
            config,
            metadata,
            tensors,
            mmap,
            tensor_data_offset,
        })
    }
}

/// Read a GGUF string: u64 length prefix followed by UTF-8 bytes (no null terminator).
fn read_gguf_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| FractureError::GgufParse(format!("invalid UTF-8 in string: {e}")))
}

fn read_metadata_value(cursor: &mut Cursor<&[u8]>) -> Result<MetadataValue> {
    let type_code = cursor.read_u32::<LittleEndian>()?;
    let typ = GgufMetadataType::from_u32(type_code)?;
    read_typed_value(cursor, typ)
}

fn read_typed_value(cursor: &mut Cursor<&[u8]>, typ: GgufMetadataType) -> Result<MetadataValue> {
    match typ {
        GgufMetadataType::Uint8 => Ok(MetadataValue::Uint8(cursor.read_u8()?)),
        GgufMetadataType::Int8 => Ok(MetadataValue::Int8(cursor.read_i8()?)),
        GgufMetadataType::Uint16 => Ok(MetadataValue::Uint16(cursor.read_u16::<LittleEndian>()?)),
        GgufMetadataType::Int16 => Ok(MetadataValue::Int16(cursor.read_i16::<LittleEndian>()?)),
        GgufMetadataType::Uint32 => Ok(MetadataValue::Uint32(cursor.read_u32::<LittleEndian>()?)),
        GgufMetadataType::Int32 => Ok(MetadataValue::Int32(cursor.read_i32::<LittleEndian>()?)),
        GgufMetadataType::Float32 => {
            Ok(MetadataValue::Float32(cursor.read_f32::<LittleEndian>()?))
        }
        GgufMetadataType::Bool => Ok(MetadataValue::Bool(cursor.read_u8()? != 0)),
        GgufMetadataType::String => Ok(MetadataValue::String(read_gguf_string(cursor)?)),
        GgufMetadataType::Uint64 => Ok(MetadataValue::Uint64(cursor.read_u64::<LittleEndian>()?)),
        GgufMetadataType::Int64 => Ok(MetadataValue::Int64(cursor.read_i64::<LittleEndian>()?)),
        GgufMetadataType::Float64 => {
            Ok(MetadataValue::Float64(cursor.read_f64::<LittleEndian>()?))
        }
        GgufMetadataType::Array => {
            let elem_type_code = cursor.read_u32::<LittleEndian>()?;
            let elem_type = GgufMetadataType::from_u32(elem_type_code)?;
            let len = cursor.read_u64::<LittleEndian>()? as usize;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(read_typed_value(cursor, elem_type)?);
            }
            Ok(MetadataValue::Array(values))
        }
    }
}

fn get_alignment(metadata: &HashMap<String, MetadataValue>) -> usize {
    metadata
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .map(|v| v as usize)
        .unwrap_or(32)
}

fn align_offset(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

/// Extract ModelConfig from GGUF metadata using standard llama.cpp key names.
fn extract_model_config(metadata: &HashMap<String, MetadataValue>) -> Result<ModelConfig> {
    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    let get_u32 = |key: &str| -> Result<usize> {
        metadata
            .get(key)
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .ok_or_else(|| FractureError::GgufParse(format!("missing metadata key: {key}")))
    };

    let get_f64 = |key: &str, default: f64| -> f64 {
        metadata
            .get(key)
            .and_then(|v| v.as_f64())
            .unwrap_or(default)
    };

    let hidden_size = get_u32(&format!("{arch}.embedding_length"))?;
    let num_layers = get_u32(&format!("{arch}.block_count"))?;
    let num_q_heads = get_u32(&format!("{arch}.attention.head_count"))?;
    let num_kv_heads = get_u32(&format!("{arch}.attention.head_count_kv"))?;
    let intermediate_size = get_u32(&format!("{arch}.feed_forward_length"))?;

    let vocab_size = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            MetadataValue::Array(arr) => Some(arr.len()),
            _ => None,
        })
        .or_else(|| {
            metadata
                .get(&format!("{arch}.vocab_size"))
                .and_then(|v| v.as_u32())
                .map(|v| v as usize)
        })
        .ok_or_else(|| FractureError::GgufParse("cannot determine vocab_size".into()))?;

    let head_dim = hidden_size / num_q_heads;
    let rope_theta = get_f64(&format!("{arch}.rope.freq_base"), 500000.0);
    let rms_norm_eps = get_f64(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5);

    let context_length = metadata
        .get(&format!("{arch}.context_length"))
        .and_then(|v| v.as_u32())
        .map(|v| v as usize)
        .unwrap_or(8192);

    let config = ModelConfig {
        hidden_size,
        num_layers,
        num_q_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        rope_theta,
        rms_norm_eps,
        max_seq_len: context_length,
    };

    config.validate()?;

    tracing::info!(
        "model config: {}L, d={}, {}Qh/{}KVh, ffn={}, vocab={}, ctx={}",
        config.num_layers,
        config.hidden_size,
        config.num_q_heads,
        config.num_kv_heads,
        config.intermediate_size,
        config.vocab_size,
        config.max_seq_len,
    );

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Write;

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        buf.write_all(s.as_bytes()).unwrap();
    }

    fn write_metadata_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
        write_gguf_string(buf, key);
        buf.write_u32::<LittleEndian>(GgufMetadataType::Uint32 as u32)
            .unwrap();
        buf.write_u32::<LittleEndian>(val).unwrap();
    }

    fn write_metadata_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
        write_gguf_string(buf, key);
        buf.write_u32::<LittleEndian>(GgufMetadataType::Float32 as u32)
            .unwrap();
        buf.write_f32::<LittleEndian>(val).unwrap();
    }

    fn write_metadata_kv_string(buf: &mut Vec<u8>, key: &str, val: &str) {
        write_gguf_string(buf, key);
        buf.write_u32::<LittleEndian>(GgufMetadataType::String as u32)
            .unwrap();
        write_gguf_string(buf, val);
    }

    fn write_metadata_kv_string_array(buf: &mut Vec<u8>, key: &str, vals: &[&str]) {
        write_gguf_string(buf, key);
        buf.write_u32::<LittleEndian>(GgufMetadataType::Array as u32)
            .unwrap();
        buf.write_u32::<LittleEndian>(GgufMetadataType::String as u32)
            .unwrap();
        buf.write_u64::<LittleEndian>(vals.len() as u64).unwrap();
        for s in vals {
            write_gguf_string(buf, s);
        }
    }

    /// Build a minimal valid GGUF v3 file in memory with one FP16 tensor.
    fn build_test_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(8).unwrap(); // metadata_kv_count

        // Metadata
        write_metadata_kv_string(&mut buf, "general.architecture", "llama");
        write_metadata_kv_u32(&mut buf, "llama.embedding_length", 64);
        write_metadata_kv_u32(&mut buf, "llama.block_count", 2);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
        write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", 128);
        write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
        // vocab via token array (4 dummy tokens)
        write_metadata_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &["a", "b", "c", "d"]);

        // Tensor info: token_embd [vocab=4, hidden=64] after reversal.
        // Write in GGUF order (innermost first): [64, 4].
        write_gguf_string(&mut buf, "token_embd.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // ndims
        buf.write_u64::<LittleEndian>(64).unwrap(); // innermost dim (hidden)
        buf.write_u64::<LittleEndian>(4).unwrap(); // outermost dim (vocab)
        buf.write_u32::<LittleEndian>(1).unwrap(); // dtype = FP16
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset (from tensor data start)

        // Pad to 32-byte alignment
        let alignment = 32;
        let current = buf.len();
        let aligned = align_offset(current, alignment);
        buf.resize(aligned, 0);

        // Tensor data: 4 * 64 * 2 bytes = 512 bytes of zeros
        buf.extend(vec![0u8; 4 * 64 * 2]);

        buf
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = build_test_gguf();
        // Corrupt magic
        data[0] = 0x00;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, &data).unwrap();
        let err = GgufParser::parse(&path).unwrap_err();
        assert!(err.to_string().contains("invalid magic"));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = build_test_gguf();
        // Set version to 2
        data[4] = 2;
        data[5] = 0;
        data[6] = 0;
        data[7] = 0;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.gguf");
        std::fs::write(&path, &data).unwrap();
        let err = GgufParser::parse(&path).unwrap_err();
        assert!(err.to_string().contains("unsupported GGUF version"));
    }

    #[test]
    fn test_parse_valid_gguf() {
        let data = build_test_gguf();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        std::fs::write(&path, &data).unwrap();

        let gguf = GgufParser::parse(&path).unwrap();

        // Config
        assert_eq!(gguf.config.hidden_size, 64);
        assert_eq!(gguf.config.num_layers, 2);
        assert_eq!(gguf.config.num_q_heads, 4);
        assert_eq!(gguf.config.num_kv_heads, 2);
        assert_eq!(gguf.config.head_dim, 16); // 64 / 4
        assert_eq!(gguf.config.intermediate_size, 128);
        assert_eq!(gguf.config.vocab_size, 4);
        assert_eq!(gguf.config.rope_theta, 500000.0);
        assert!((gguf.config.rms_norm_eps - 1e-5).abs() < 1e-10);

        // Tensors
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "token_embd.weight");
        assert_eq!(gguf.tensors[0].shape, vec![4, 64]);
        assert_eq!(gguf.tensors[0].dtype, DType::FP16);
    }

    #[test]
    fn test_tensor_data_accessible() {
        let data = build_test_gguf();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        std::fs::write(&path, &data).unwrap();

        let gguf = GgufParser::parse(&path).unwrap();
        let t = &gguf.tensors[0];
        let start = gguf.tensor_data_offset + t.offset as usize;
        let size = t.shape.iter().product::<usize>() * t.dtype.size_bytes();
        assert!(start + size <= gguf.mmap.len());
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }

    #[test]
    fn test_missing_embedding_length() {
        let mut buf = Vec::new();

        // Header
        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(2).unwrap(); // metadata_kv_count

        // Only general.architecture and block_count, no embedding_length
        write_metadata_kv_string(&mut buf, "general.architecture", "llama");
        write_metadata_kv_u32(&mut buf, "llama.block_count", 2);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_embed.gguf");
        std::fs::write(&path, &buf).unwrap();

        let err = GgufParser::parse(&path).unwrap_err();
        assert!(
            err.to_string().contains("embedding_length"),
            "expected 'embedding_length' in error: {err}"
        );
    }

    #[test]
    fn test_vocab_size_from_arch_key() {
        let mut buf = Vec::new();

        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(8).unwrap(); // metadata_kv_count

        write_metadata_kv_string(&mut buf, "general.architecture", "llama");
        write_metadata_kv_u32(&mut buf, "llama.embedding_length", 64);
        write_metadata_kv_u32(&mut buf, "llama.block_count", 2);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
        write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", 128);
        write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
        // Use arch key for vocab_size instead of tokenizer.ggml.tokens
        write_metadata_kv_u32(&mut buf, "llama.vocab_size", 100);

        // One tensor [100, 64] FP16
        write_gguf_string(&mut buf, "token_embd.weight");
        buf.write_u32::<LittleEndian>(2).unwrap();
        buf.write_u64::<LittleEndian>(100).unwrap();
        buf.write_u64::<LittleEndian>(64).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap(); // FP16
        buf.write_u64::<LittleEndian>(0).unwrap();

        let alignment = 32;
        let current = buf.len();
        let aligned = align_offset(current, alignment);
        buf.resize(aligned, 0);
        buf.extend(vec![0u8; 100 * 64 * 2]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab_arch.gguf");
        std::fs::write(&path, &buf).unwrap();

        let gguf = GgufParser::parse(&path).unwrap();
        assert_eq!(gguf.config.vocab_size, 100);
    }

    #[test]
    fn test_parse_multiple_tensors() {
        let mut buf = Vec::new();

        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
        buf.write_u64::<LittleEndian>(3).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(8).unwrap(); // metadata_kv_count

        write_metadata_kv_string(&mut buf, "general.architecture", "llama");
        write_metadata_kv_u32(&mut buf, "llama.embedding_length", 64);
        write_metadata_kv_u32(&mut buf, "llama.block_count", 2);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
        write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", 128);
        write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
        write_metadata_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &["a", "b", "c", "d"]);

        // Tensor 0: after reversal [4, 64]. GGUF order: [64, 4].
        write_gguf_string(&mut buf, "tensor_a");
        buf.write_u32::<LittleEndian>(2).unwrap();
        buf.write_u64::<LittleEndian>(64).unwrap();
        buf.write_u64::<LittleEndian>(4).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap(); // FP16
        buf.write_u64::<LittleEndian>(0).unwrap();

        // Tensor 1: [10] FP32 at offset 512 (1D, no reversal)
        write_gguf_string(&mut buf, "tensor_b");
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.write_u64::<LittleEndian>(10).unwrap();
        buf.write_u32::<LittleEndian>(0).unwrap(); // FP32
        buf.write_u64::<LittleEndian>(512).unwrap();

        // Tensor 2: after reversal [2, 3, 4]. GGUF order: [4, 3, 2].
        write_gguf_string(&mut buf, "tensor_c");
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(4).unwrap();
        buf.write_u64::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(2).unwrap();
        buf.write_u32::<LittleEndian>(30).unwrap(); // BF16
        buf.write_u64::<LittleEndian>(552).unwrap();

        let alignment = 32;
        let current = buf.len();
        let aligned = align_offset(current, alignment);
        buf.resize(aligned, 0);

        // Tensor data: 512 (tensor_a) + 40 (tensor_b) + 48 (tensor_c) = 600 total, but
        // tensor_c starts at 552 so we need 552 + 48 = 600 bytes
        buf.extend(vec![0u8; 600]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.gguf");
        std::fs::write(&path, &buf).unwrap();

        let gguf = GgufParser::parse(&path).unwrap();
        assert_eq!(gguf.tensors.len(), 3);

        assert_eq!(gguf.tensors[0].name, "tensor_a");
        assert_eq!(gguf.tensors[0].shape, vec![4, 64]);
        assert_eq!(gguf.tensors[0].dtype, DType::FP16);

        assert_eq!(gguf.tensors[1].name, "tensor_b");
        assert_eq!(gguf.tensors[1].shape, vec![10]);
        assert_eq!(gguf.tensors[1].dtype, DType::FP32);

        assert_eq!(gguf.tensors[2].name, "tensor_c");
        assert_eq!(gguf.tensors[2].shape, vec![2, 3, 4]);
        assert_eq!(gguf.tensors[2].dtype, DType::BF16);
    }

    #[test]
    fn test_missing_vocab_size() {
        // Build a GGUF with neither tokenizer.ggml.tokens nor llama.vocab_size.
        // The parser should fail with an error about vocab_size.
        let mut buf = Vec::new();

        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(7).unwrap(); // metadata_kv_count

        write_metadata_kv_string(&mut buf, "general.architecture", "llama");
        write_metadata_kv_u32(&mut buf, "llama.embedding_length", 64);
        write_metadata_kv_u32(&mut buf, "llama.block_count", 2);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
        write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", 128);
        write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
        // No tokenizer.ggml.tokens, no llama.vocab_size

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_vocab.gguf");
        std::fs::write(&path, &buf).unwrap();

        let err = GgufParser::parse(&path).unwrap_err();
        assert!(
            matches!(err, FractureError::GgufParse(_)),
            "expected GgufParse error, got: {err}"
        );
        assert!(
            err.to_string().contains("vocab_size"),
            "expected error about vocab_size, got: {err}"
        );
    }

    #[test]
    fn test_tensor_info_multiple_types() {
        // Verify that FP16 and FP32 tensors have correct shapes and dtypes parsed.
        // This enhances test_parse_multiple_tensors by also checking FP16 vs FP32 size_bytes.
        let mut buf = Vec::new();

        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
        buf.write_u64::<LittleEndian>(2).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(8).unwrap(); // metadata_kv_count

        write_metadata_kv_string(&mut buf, "general.architecture", "llama");
        write_metadata_kv_u32(&mut buf, "llama.embedding_length", 64);
        write_metadata_kv_u32(&mut buf, "llama.block_count", 2);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
        write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", 128);
        write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
        write_metadata_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &["a", "b", "c", "d"]);

        // Tensor 0: after reversal [8, 16]. GGUF order: [16, 8].
        write_gguf_string(&mut buf, "fp16_tensor");
        buf.write_u32::<LittleEndian>(2).unwrap();
        buf.write_u64::<LittleEndian>(16).unwrap();
        buf.write_u64::<LittleEndian>(8).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap(); // FP16
        buf.write_u64::<LittleEndian>(0).unwrap();

        // Tensor 1: after reversal [4, 8]. GGUF order: [8, 4].
        write_gguf_string(&mut buf, "fp32_tensor");
        buf.write_u32::<LittleEndian>(2).unwrap();
        buf.write_u64::<LittleEndian>(8).unwrap();
        buf.write_u64::<LittleEndian>(4).unwrap();
        buf.write_u32::<LittleEndian>(0).unwrap(); // FP32
        buf.write_u64::<LittleEndian>(256).unwrap();

        let alignment = 32;
        let current = buf.len();
        let aligned = align_offset(current, alignment);
        buf.resize(aligned, 0);

        // Tensor data: 256 (fp16) + 128 (fp32: 4*8*4) = 384 bytes
        buf.extend(vec![0u8; 384]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi_dtype.gguf");
        std::fs::write(&path, &buf).unwrap();

        let gguf = GgufParser::parse(&path).unwrap();
        assert_eq!(gguf.tensors.len(), 2);

        // FP16 tensor
        assert_eq!(gguf.tensors[0].name, "fp16_tensor");
        assert_eq!(gguf.tensors[0].shape, vec![8, 16]);
        assert_eq!(gguf.tensors[0].dtype, DType::FP16);
        assert_eq!(gguf.tensors[0].dtype.size_bytes(), 2);

        // FP32 tensor
        assert_eq!(gguf.tensors[1].name, "fp32_tensor");
        assert_eq!(gguf.tensors[1].shape, vec![4, 8]);
        assert_eq!(gguf.tensors[1].dtype, DType::FP32);
        assert_eq!(gguf.tensors[1].dtype.size_bytes(), 4);
    }

    #[test]
    fn test_unsupported_dtype_q4_0() {
        let mut buf = Vec::new();

        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(8).unwrap(); // metadata_kv_count

        write_metadata_kv_string(&mut buf, "general.architecture", "llama");
        write_metadata_kv_u32(&mut buf, "llama.embedding_length", 64);
        write_metadata_kv_u32(&mut buf, "llama.block_count", 2);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
        write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", 128);
        write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
        write_metadata_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &["a", "b", "c", "d"]);

        // Tensor with dtype code 2 (Q4_0)
        write_gguf_string(&mut buf, "quantized.weight");
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.write_u64::<LittleEndian>(64).unwrap();
        buf.write_u32::<LittleEndian>(2).unwrap(); // Q4_0
        buf.write_u64::<LittleEndian>(0).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q4_0.gguf");
        std::fs::write(&path, &buf).unwrap();

        let err = GgufParser::parse(&path).unwrap_err();
        assert!(
            matches!(err, FractureError::UnsupportedDType(_)),
            "expected UnsupportedDType, got: {err}"
        );
    }

    #[test]
    fn test_rms_norm_eps_extraction() {
        // Build a GGUF with explicit rms_norm_eps = 1e-6 and verify it is extracted.
        let mut buf = Vec::new();

        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(9).unwrap(); // metadata_kv_count (8 base + 1 eps)

        write_metadata_kv_string(&mut buf, "general.architecture", "llama");
        write_metadata_kv_u32(&mut buf, "llama.embedding_length", 64);
        write_metadata_kv_u32(&mut buf, "llama.block_count", 2);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
        write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", 128);
        write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
        write_metadata_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &["a", "b", "c", "d"]);
        // Explicit rms_norm_eps
        write_metadata_kv_f32(
            &mut buf,
            "llama.attention.layer_norm_rms_epsilon",
            1e-6,
        );

        // One tensor [4, 64] FP16
        write_gguf_string(&mut buf, "token_embd.weight");
        buf.write_u32::<LittleEndian>(2).unwrap();
        buf.write_u64::<LittleEndian>(64).unwrap();
        buf.write_u64::<LittleEndian>(4).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap(); // FP16
        buf.write_u64::<LittleEndian>(0).unwrap();

        let current = buf.len();
        let aligned = align_offset(current, 32);
        buf.resize(aligned, 0);
        buf.extend(vec![0u8; 4 * 64 * 2]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eps.gguf");
        std::fs::write(&path, &buf).unwrap();

        let gguf = GgufParser::parse(&path).unwrap();
        assert!(
            (gguf.config.rms_norm_eps - 1e-6).abs() < 1e-10,
            "expected rms_norm_eps ~1e-6, got {}",
            gguf.config.rms_norm_eps
        );
    }

    #[test]
    fn test_context_length_extraction() {
        // Test 1: explicit context_length = 4096
        let mut buf = Vec::new();

        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(9).unwrap(); // metadata_kv_count (8 base + 1 ctx)

        write_metadata_kv_string(&mut buf, "general.architecture", "llama");
        write_metadata_kv_u32(&mut buf, "llama.embedding_length", 64);
        write_metadata_kv_u32(&mut buf, "llama.block_count", 2);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
        write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
        write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", 128);
        write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
        write_metadata_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &["a", "b", "c", "d"]);
        // Explicit context_length
        write_metadata_kv_u32(&mut buf, "llama.context_length", 4096);

        write_gguf_string(&mut buf, "token_embd.weight");
        buf.write_u32::<LittleEndian>(2).unwrap();
        buf.write_u64::<LittleEndian>(64).unwrap();
        buf.write_u64::<LittleEndian>(4).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap();

        let current = buf.len();
        let aligned = align_offset(current, 32);
        buf.resize(aligned, 0);
        buf.extend(vec![0u8; 4 * 64 * 2]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ctx.gguf");
        std::fs::write(&path, &buf).unwrap();

        let gguf = GgufParser::parse(&path).unwrap();
        assert_eq!(
            gguf.config.max_seq_len, 4096,
            "expected max_seq_len=4096, got {}",
            gguf.config.max_seq_len
        );

        // Test 2: default context_length when key is absent (should be 8192)
        let data = build_test_gguf(); // has no context_length key
        let dir2 = tempfile::tempdir().unwrap();
        let path2 = dir2.path().join("default_ctx.gguf");
        std::fs::write(&path2, &data).unwrap();

        let gguf2 = GgufParser::parse(&path2).unwrap();
        assert_eq!(
            gguf2.config.max_seq_len, 8192,
            "expected default max_seq_len=8192, got {}",
            gguf2.config.max_seq_len
        );
    }
}
