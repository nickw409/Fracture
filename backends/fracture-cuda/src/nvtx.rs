//! NVTX marker wrappers for Nsight Systems profiling.

use crate::ffi;
use std::ffi::CString;

/// Push a named NVTX range. Shows up in Nsight Systems timeline.
pub fn range_push(name: &str) {
    let c_name = CString::new(name).unwrap_or_else(|_| CString::new("invalid").unwrap());
    unsafe { ffi::nvtxRangePushA(c_name.as_ptr()) };
}

/// Pop the most recent NVTX range.
pub fn range_pop() {
    unsafe { ffi::nvtxRangePop() };
}
