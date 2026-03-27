//! CUDA event-based GPU timers.

use crate::ffi::*;
use fracture_core::{DeviceTimer, FractureError, Result};
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Mutex;

/// Manages CUDA event pairs for GPU timing.
pub struct TimerManager {
    timers: Mutex<TimerState>,
}

struct TimerState {
    events: HashMap<u64, (cudaEvent_t, cudaEvent_t)>,
    next_id: u64,
}

// CUDA event handles are safe to use from any thread when synchronized.
unsafe impl Send for TimerState {}

impl TimerManager {
    pub fn new() -> Self {
        Self {
            timers: Mutex::new(TimerState {
                events: HashMap::new(),
                next_id: 1,
            }),
        }
    }

    pub fn create(&self) -> Result<DeviceTimer> {
        let mut start: cudaEvent_t = std::ptr::null_mut();
        let mut stop: cudaEvent_t = std::ptr::null_mut();

        let err = unsafe { cudaEventCreate(&mut start) };
        if err != CUDA_SUCCESS {
            let msg = unsafe { CStr::from_ptr(cudaGetErrorString(err)) };
            return Err(FractureError::Backend(format!(
                "cudaEventCreate failed: {}", msg.to_string_lossy()
            )));
        }

        let err = unsafe { cudaEventCreate(&mut stop) };
        if err != CUDA_SUCCESS {
            unsafe { cudaEventDestroy(start) };
            let msg = unsafe { CStr::from_ptr(cudaGetErrorString(err)) };
            return Err(FractureError::Backend(format!(
                "cudaEventCreate failed: {}", msg.to_string_lossy()
            )));
        }

        let mut state = self.timers.lock().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        state.events.insert(id, (start, stop));

        Ok(DeviceTimer(id))
    }

    pub fn start(&self, timer: &DeviceTimer, stream: cudaStream_t) -> Result<()> {
        let state = self.timers.lock().unwrap();
        let (start, _) = state.events.get(&timer.0)
            .ok_or_else(|| FractureError::Backend(format!("timer {} not found", timer.0)))?;

        let err = unsafe { cudaEventRecord(*start, stream) };
        if err != CUDA_SUCCESS {
            let msg = unsafe { CStr::from_ptr(cudaGetErrorString(err)) };
            return Err(FractureError::Backend(format!(
                "cudaEventRecord failed: {}", msg.to_string_lossy()
            )));
        }
        Ok(())
    }

    pub fn stop(&self, timer: &DeviceTimer, stream: cudaStream_t) -> Result<f32> {
        let state = self.timers.lock().unwrap();
        let (start, stop) = state.events.get(&timer.0)
            .ok_or_else(|| FractureError::Backend(format!("timer {} not found", timer.0)))?;

        let err = unsafe { cudaEventRecord(*stop, stream) };
        if err != CUDA_SUCCESS {
            let msg = unsafe { CStr::from_ptr(cudaGetErrorString(err)) };
            return Err(FractureError::Backend(format!(
                "cudaEventRecord failed: {}", msg.to_string_lossy()
            )));
        }

        let err = unsafe { cudaEventSynchronize(*stop) };
        if err != CUDA_SUCCESS {
            let msg = unsafe { CStr::from_ptr(cudaGetErrorString(err)) };
            return Err(FractureError::Backend(format!(
                "cudaEventSynchronize failed: {}", msg.to_string_lossy()
            )));
        }

        let mut elapsed: f32 = 0.0;
        let err = unsafe { cudaEventElapsedTime(&mut elapsed, *start, *stop) };
        if err != CUDA_SUCCESS {
            let msg = unsafe { CStr::from_ptr(cudaGetErrorString(err)) };
            return Err(FractureError::Backend(format!(
                "cudaEventElapsedTime failed: {}", msg.to_string_lossy()
            )));
        }

        Ok(elapsed)
    }

    pub fn destroy(&self, timer: &DeviceTimer) -> Result<()> {
        let mut state = self.timers.lock().unwrap();
        let (start, stop) = state.events.remove(&timer.0)
            .ok_or_else(|| FractureError::Backend(format!("timer {} not found", timer.0)))?;

        unsafe {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        Ok(())
    }
}

impl Drop for TimerManager {
    fn drop(&mut self) {
        let state = self.timers.lock().unwrap_or_else(|e| e.into_inner());
        for (_, (start, stop)) in state.events.iter() {
            unsafe {
                cudaEventDestroy(*start);
                cudaEventDestroy(*stop);
            }
        }
    }
}
