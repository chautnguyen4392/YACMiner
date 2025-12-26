# System RAM Support for padbuffer8_RAM Buffers

## Overview

This document describes the implementation of system RAM support for additional `padbuffer8_RAM` buffers in YACMiner. This feature allows the miner to utilize system RAM in addition to GPU VRAM for storing scrypt lookup tables, enabling larger thread concurrency values when VRAM is limited.

## Key Features

- **New Command Line Option**: `--use-system-ram` (disabled by default)
- **Maximum Buffers**: Up to 2 `padbuffer8_RAM` buffers per GPU
- **Memory Distribution**: System RAM is distributed equally among all GPUs
- **Priority**: VRAM is fully utilized first, then system RAM
- **Constraints**: Each buffer size must not exceed `cgpu->max_alloc`

## Implementation Details

### 1. Command Line Option

**Location**: `yacminer.c`

- **Option**: `--use-system-ram`
- **Type**: Boolean flag (disabled by default)
- **Description**: Enables creation of additional `padbuffer8_RAM` buffers using system RAM
- **Variable**: `opt_use_system_ram`

### 2. Data Structures

**Location**: `ocl.h`

Added to `_clState` structure:
```c
cl_mem padbuffer8_RAM[2];              // System RAM buffers (up to 2)
size_t num_padbuffers_RAM;              // Number of padbuffer8_RAM buffers (0-2)
size_t groups_per_buffer_RAM[2];       // Number of groups per buffer for system RAM
```

### 3. System RAM Calculation

**Location**: `ocl.c` - `get_available_system_ram_per_gpu()`

This function calculates available system RAM per GPU:

1. **Primary Method**: Reads from `/proc/meminfo`
   - Extracts `MemTotal`, `MemFree`, and `MemAvailable`
   - Logs all three values
   - Uses `MemAvailable` if available, otherwise falls back to `MemFree`

2. **Fallback Method**: Uses `sysinfo()` system call
   - Calculates available memory as `freeram + bufferram`
   - Logs `MemTotal`, `MemFree`, and calculated available memory

3. **Distribution**: 
   - Counts total number of GPUs using `clDevicesNum()`
   - Distributes available RAM equally: `ram_per_gpu = mem_available / num_gpus`
   - Logs the distribution per GPU

**Key Points**:
- Returns 0 if system memory information cannot be obtained
- All memory values are converted from KB to bytes
- Logs comprehensive memory information for debugging

### 4. Memory Validation

**Location**: `ocl.c` - `initCl()` function

When `--use-system-ram` is enabled:

1. **Calculate Available System RAM**:
   - Calls `get_available_system_ram_per_gpu()` to get RAM per GPU
   - If calculation fails, disables the option and logs error

2. **Total Memory Check**:
   - Calculates `total_available_mem = remaining_mem_size + available_system_ram`
   - Validates that `total_groups_size <= total_available_mem`
   - If exceeded, logs detailed error and returns NULL

3. **Error Messages**:
   - Shows VRAM and system RAM breakdown when enabled
   - Shows VRAM details when disabled
   - Includes `global_mem_size` and `other_buffers_size` for context

### 5. Buffer Optimization Logic

**Location**: `ocl.c` - `initCl()` function (after VRAM buffer optimization)

The optimization process:

1. **Calculate Remaining Groups**:
   - Determines how many groups are already covered by VRAM buffers
   - Calculates `remaining_groups = number_groups - groups_covered_by_vram`

2. **Single Buffer Optimization**:
   - Tries 1 buffer configuration first
   - Checks: `single_ram_buffer_size <= available_system_ram && single_ram_buffer_size <= cgpu->max_alloc`
   - If utilization >= 99% of available RAM, uses this configuration

3. **Two Buffer Optimization**:
   - Tries all possible splits of remaining groups between 2 buffers
   - For each split, checks:
     - `ram1_size <= cgpu->max_alloc`
     - `ram2_size <= cgpu->max_alloc`
     - `total_ram_size <= available_system_ram`
   - Selects configuration with maximum utilization

4. **Selection Criteria**:
   - Target: Use at least 99% of available system RAM
   - Prefers fewer buffers if target is met
   - Selects configuration with best utilization

**Key Constraints**:
- Each individual buffer size must not exceed `cgpu->max_alloc`
- Total size of all buffers must not exceed `available_system_ram`
- Maximum 2 buffers allowed

### 6. Compiler Options

**Location**: `ocl.c` - `initCl()` function

Passes buffer configuration to OpenCL kernel via compiler defines:

```c
-D NUM_PADBUFFERS_RAM=<num_buffers>
-D THREADS_PER_BUFFER_RAM_0=<threads_buffer0>
-D THREADS_PER_BUFFER_RAM_1=<threads_buffer1>
```

Where:
- `threads_bufferN = groups_per_buffer_RAM[N] * wsize`

### 7. Buffer Creation

**Location**: `ocl.c` - `initCl()` function

Creates `padbuffer8_RAM` buffers using `CL_MEM_ALLOC_HOST_PTR`:

1. **Pre-creation Validation**:
   - Checks each buffer size against `cgpu->max_alloc`
   - If any buffer exceeds limit, logs error, releases buffers, and returns NULL

2. **Buffer Creation**:
   - Uses `CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR` flags
   - Creates buffers sequentially
   - If creation fails, logs error and exits YACMiner (no fallback)

3. **Error Handling**:
   - Releases all previously created buffers (both VRAM and system RAM)
   - Calls `quit()` to exit YACMiner on failure

### 8. Kernel Integration

#### 8.1 Host Code (driver-opencl.c)

**Monolithic Kernel** (`queue_scrypt_kernel`):
- Passes all `padbuffer8` buffers (VRAM) first
- Then passes all `padbuffer8_RAM` buffers (system RAM)
- Order: `padbuffer8[0..N-1]`, then `padbuffer8_RAM[0..M-1]`

**Split Kernel Part 2** (`opencl_scanhash`):
- Same order: VRAM buffers first, then system RAM buffers
- All buffers passed as kernel arguments

**Cleanup** (`opencl_thread_shutdown`):
- Releases all `padbuffer8_RAM` buffers
- Logs number of buffers released

#### 8.2 Kernel Code (scrypt-chacha.cl)

**Kernel Signatures**:
- `search84`: Accepts `padcache_ram0` and `padcache_ram1` parameters (conditionally)
- `search84_part2`: Same parameters for split kernel execution

**Buffer Selection Logic**:

1. **Calculate Total VRAM Threads**:
   ```c
   uint total_vram_threads = THREADS_PER_BUFFER_0 + THREADS_PER_BUFFER_1 + THREADS_PER_BUFFER_2;
   ```

2. **Check VRAM Buffers First**:
   - Uses existing logic to select from `padcache0`, `padcache1`, `padcache2`
   - Adjusts `relative_gid` for buffer-relative indexing

3. **Check System RAM Buffers**:
   - Only if `padcache == NULL` and `relative_gid >= total_vram_threads`
   - Calculates `ram_relative_gid = relative_gid - total_vram_threads`
   - Selects buffer based on `ram_relative_gid`:
     - If `NUM_PADBUFFERS_RAM == 1`: Uses `padcache_ram0`
     - If `NUM_PADBUFFERS_RAM == 2`: 
       - If `ram_relative_gid < THREADS_PER_BUFFER_RAM_0`: Uses `padcache_ram0`
       - Else: Uses `padcache_ram1`, adjusts `relative_gid`

4. **ROMix Call**:
   - Uses selected buffer and adjusted `relative_gid`
   - `buffer_xSIZE` ensures correct buffer-relative indexing

**Key Points**:
- `relative_gid` is calculated using `get_group_id(0) * WORKSIZE + get_local_id(0)`
- This correctly handles `global_work_offset` because group/local IDs are relative to work batch
- Buffer selection prioritizes VRAM, then system RAM
- Each buffer uses its own `buffer_xSIZE` for correct indexing

## Memory Layout

```
┌─────────────────────────────────────┐
│         VRAM Buffers                │
│  padbuffer8[0]  padbuffer8[1] ...   │
│  (Groups 0..N-1)                    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      System RAM Buffers              │
│  padbuffer8_RAM[0]  padbuffer8_RAM[1]│
│  (Groups N..M-1)                     │
└─────────────────────────────────────┘
```

Thread distribution:
- Threads 0 to (total_vram_threads - 1): Use VRAM buffers
- Threads total_vram_threads to (total_threads - 1): Use system RAM buffers

## Usage Example

```bash
# Enable system RAM support
./yacminer --scrypt-chacha-84 --use-system-ram

# The miner will:
# 1. Calculate available system RAM per GPU
# 2. Optimize VRAM buffers first
# 3. Optimize system RAM buffers for remaining groups
# 4. Create all buffers and start mining
```

## Logging

The implementation provides comprehensive logging:

- **System RAM Info**: MemTotal, MemFree, MemAvailable (or calculated available)
- **Distribution**: RAM per GPU and total GPU count
- **Buffer Configuration**: Number of buffers and groups per buffer
- **Memory Utilization**: Percentage of available memory used
- **Buffer Creation**: Success/failure for each buffer
- **Error Messages**: Detailed information when memory constraints are violated

## Error Handling

1. **System RAM Calculation Failure**:
   - Logs error and disables `opt_use_system_ram`
   - Continues with VRAM-only operation

2. **Memory Exceeded**:
   - Logs detailed error showing required vs available memory
   - Returns NULL from `initCl()`, causing GPU initialization to fail

3. **Buffer Creation Failure**:
   - Logs error with buffer index and size
   - Releases all created buffers
   - Calls `quit()` to exit YACMiner

4. **Buffer Size Validation**:
   - Checks each buffer against `max_alloc` before creation
   - Prevents OpenCL buffer creation failures

## Performance Considerations

- **System RAM Access**: Slower than VRAM, but enables larger thread concurrency
- **Memory Bandwidth**: System RAM bandwidth is shared with CPU and other processes
- **Pinned Memory**: Uses `CL_MEM_ALLOC_HOST_PTR` for better transfer performance
- **Buffer Distribution**: Equal distribution among GPUs may not be optimal for all systems

## Future Improvements

Potential enhancements:
- Configurable RAM allocation per GPU (not just equal distribution)
- Dynamic buffer size adjustment based on runtime performance
- Support for more than 2 system RAM buffers (currently limited to 2)
- Memory usage monitoring and reporting

