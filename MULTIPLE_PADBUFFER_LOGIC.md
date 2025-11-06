# Multiple Padbuffer8 Buffers - Implementation Summary

## Overview

This feature implements support for multiple `padbuffer8` buffers (up to 3) to better utilize available GPU global memory when the maximum allocation size (`max_alloc`) is less than the total global memory size. This allows the miner to use more memory efficiently, potentially improving performance by allowing larger thread concurrency settings.

## Problem Statement

Previously, the miner could only create a single `padbuffer8` buffer limited by `CL_DEVICE_MAX_MEM_ALLOC_SIZE`. However, many GPUs have significantly more total global memory (`CL_DEVICE_GLOBAL_MEM_SIZE`) than the maximum single allocation size. This meant that a large portion of available memory was unused, limiting the maximum thread concurrency.

## Solution

By creating multiple smaller buffers (each ≤ `max_alloc`) that together utilize nearly all available global memory, we can:
- Maximize memory utilization
- Support higher thread concurrency values
- Improve mining performance

## Algorithm

### 1. Memory Source Detection

The algorithm first determines the available GPU memory using the most accurate method available:

1. **AMD Free Memory Extension** (preferred, if available):
   - Checks for `cl_amd_device_attribute_query` extension
   - Queries `CL_DEVICE_GLOBAL_FREE_MEMORY_AMD` which returns an array of 4 `size_t` values (free memory in KB)
   - Uses the first element (largest free memory block)
   - Converts from KB to bytes: `free_mem[0] * 1024`
   - More accurate for allocation decisions as it reflects actual free memory

2. **Standard Global Memory** (fallback):
   - Uses `CL_DEVICE_GLOBAL_MEM_SIZE` if AMD extension is not available or query fails
   - Represents total GPU global memory (may include allocated memory)

The selected memory source is logged for debugging purposes.

### 2. Memory Calculation

The algorithm then calculates:
- **Remaining memory**: `global_mem_size - (CLbuffer0 + outputBuffer + temp_X_buffer + temp_X2_buffer)`
- **Each group size**: `128 * ipt * work_size` where `ipt = (bsize / lookup_gap + (bsize % lookup_gap > 0))`
- **Number of groups**: `thread_concurrency / work_size`
- **Total groups size**: `number_groups * each_group_size`

### 3. Memory Safety Check

**Critical**: Before proceeding, the algorithm validates that all groups fit within available memory:

```c
if (remaining_mem_size > 0 && total_groups_size > remaining_mem_size) {
    // Log error and stop initialization
    return NULL;
}
```

This prevents memory overlap which would cause:
- GPU crashes
- Memory corruption
- Incorrect mining results

If validation fails, YACMiner logs detailed error information and stops initialization, requiring the user to reduce `thread_concurrency` or `lookup_gap`.

### 4. Buffer Selection Strategy

The algorithm uses a **greedy approach** to find the **smallest number of buffers** that maximizes memory utilization:

1. **Target**: Use at least **99% of remaining memory**
2. **Start with 1 buffer**: Check if a single buffer can meet the target
   - If yes → use 1 buffer (optimal)
   - If no → continue to 2 buffers
3. **Try 2 buffers**: Search all possible group distributions
   - Find the configuration that maximizes total memory used
   - If it meets the target → use 2 buffers (stop here)
   - If not → continue to 3 buffers
4. **Try 3 buffers**: Only if 2 buffers didn't meet target
   - Search all combinations of group distributions
   - Select the configuration that maximizes total memory used

### 5. Constraints

Each buffer configuration must satisfy:
- Each buffer size ≤ `max_alloc`
- Total size of all buffers ≤ `remaining_mem_size`
- All groups must be allocated (no groups left unassigned)

### 6. Group Distribution

For each buffer count, the algorithm:
- **2 buffers**: Tries all splits from `number_groups/2` down to 1
- **3 buffers**: Tries all combinations where:
  - `groups_buf1` ranges from `number_groups/3` down to 1
  - `groups_buf2` ranges from `(number_groups - groups_buf1)/2` down to 1
  - `groups_buf3 = number_groups - groups_buf1 - groups_buf2`

## Implementation Details

### Data Structures

**In `ocl.h` (`_clState` structure):**
```c
cl_mem padbuffer8[3];              // Array of up to 3 buffers
size_t num_padbuffers;              // Number of buffers (1-3)
size_t groups_per_buffer[3];        // Groups allocated to each buffer
```

**In `miner.h` (`cgpu_info` structure):**
```c
cl_ulong global_mem_size;           // Total GPU global memory
```

### Compiler Options

The buffer configuration is passed to the OpenCL kernel as compile-time defines:
- `NUM_PADBUFFERS`: Number of buffers (1-3)
- `THREADS_PER_BUFFER_0`: Threads in first buffer (`groups_per_buffer[0] * WORKSIZE`)
- `THREADS_PER_BUFFER_1`: Threads in second buffer (`groups_per_buffer[1] * WORKSIZE`)
- `THREADS_PER_BUFFER_2`: Threads in third buffer (`groups_per_buffer[2] * WORKSIZE`)

**Note**: Threads per buffer (not groups) are passed to kernels because `scrypt_ROMix` uses thread-based indexing (`xSIZE_override` parameter) for correct buffer-relative indexing.

### Kernel Modifications

**Optimized Buffer Selection Logic:**

All kernels (`search`, `search84`, `search84_part2`) use an optimized approach with only **4 variables**:

1. `group_id = get_group_id(0)` - Group ID relative to work batch (handles `global_work_offset` correctly)
2. `relative_gid = group_id * WORKSIZE + get_local_id(0)` - Relative thread ID within work batch
3. `padcache` - Pointer to selected buffer
4. `buffer_xSIZE` - Thread count for selected buffer (for ROMix indexing)

**Buffer Selection Algorithm:**

```c
const uint group_id = get_group_id(0);
uint relative_gid = group_id * WORKSIZE + get_local_id(0);
__global uchar * padcache = (__global uchar *)0;
uint buffer_xSIZE = 0;

#if NUM_PADBUFFERS == 1
    padcache = padcache0;
    buffer_xSIZE = THREADS_PER_BUFFER_0;
#elif NUM_PADBUFFERS == 2
    if (relative_gid < THREADS_PER_BUFFER_0) {
        padcache = padcache0;
        buffer_xSIZE = THREADS_PER_BUFFER_0;
    } else {
        padcache = padcache1;
        buffer_xSIZE = THREADS_PER_BUFFER_1;
        relative_gid = relative_gid - THREADS_PER_BUFFER_0;
    }
#elif NUM_PADBUFFERS == 3
    // Similar logic for 3 buffers
#endif

scrypt_ROMix(X, (__global uint4 *)padcache, relative_gid, buffer_xSIZE);
```

**Key Optimizations:**

1. **No `group_offset` variable**: Offset is calculated directly in the conditional blocks
2. **Direct ROMix call**: `relative_gid` is adjusted inline before calling `scrypt_ROMix`
3. **Minimal variables**: Only 4 variables needed for buffer selection and ROMix call
4. **Correct indexing**: `buffer_xSIZE` ensures ROMix uses buffer-relative indexing, preventing out-of-bounds access

**scrypt_ROMix Changes:**

- Added `xSIZE_override` parameter to override the default `CONCURRENT_THREADS` for indexing
- When `xSIZE_override > 0`, uses it instead of `CONCURRENT_THREADS` for buffer-relative indexing
- Prevents memory access faults when using multiple buffers

**Monolithic kernels (`search`, `search84`):**
- Accept multiple `padcache` parameters (padcache0, padcache1, padcache2)
- Use optimized buffer selection logic above
- Pass `buffer_xSIZE` to `scrypt_ROMix` for correct indexing

**Split kernel (`search84_part2`):**
- Same optimized buffer selection logic as monolithic kernels
- Distributes work across buffers based on `relative_gid`

### Host Code Changes

**In `driver-opencl.c`:**
- `queue_scrypt_kernel()`: Passes all buffers to monolithic kernel
- `opencl_scanhash()`: Passes all buffers to split kernel Part 2
- `opencl_thread_shutdown()`: Releases all buffers properly

## Key Features

### 1. Global Work Offset Handling

The implementation correctly handles `global_work_offset` by:
- Using `get_group_id(0)` instead of `gid / WORKSIZE` for buffer selection
- `get_group_id(0)` returns group ID relative to work batch (without offset)
- Calculating relative `gid` using `group_id * WORKSIZE + get_local_id(0)`

This ensures correct buffer selection and ROMix indexing regardless of nonce offset.

### 2. Memory Utilization Optimization

The algorithm prioritizes:
1. **Minimizing buffer count** (prefer fewer buffers)
2. **Maximizing memory usage** (target 99% utilization)
3. **Balancing group distribution** (even distribution when possible)

### 3. Fallback Behavior

If multiple buffers aren't beneficial:
- Falls back to single buffer
- Respects `max_alloc` limit
- Recalculates `number_groups` if needed

## Benefits

1. **Better Memory Utilization**: Uses nearly all available GPU memory (up to 99%)
2. **Higher Performance**: Supports larger thread concurrency values
3. **Automatic Optimization**: Automatically selects optimal buffer configuration
4. **Backward Compatible**: Falls back to single buffer when appropriate
5. **Correct Offset Handling**: Properly handles `global_work_offset` for nonce ranges
6. **Accurate Memory Detection**: Uses AMD free memory extension when available for better allocation decisions
7. **Memory Safety**: Validates memory requirements before initialization to prevent crashes
8. **Optimized Kernel Code**: Minimal variables and direct calculations for better performance
9. **Correct Buffer Indexing**: Uses buffer-relative thread counts to prevent out-of-bounds access

## Example Scenarios

### Scenario 1: Single Buffer Insufficient - Needs Multiple Buffers
- Global memory: 8 GB
- Max alloc: 2 GB (each buffer limited to 2 GB)
- Remaining: 7.5 GB
- Target utilization: 7.425 GB (99% of 7.5 GB)
- Single buffer: 2 GB max (26% utilization) ❌ Below target
- Two buffers: 2 GB + 2 GB = 4 GB total (53% utilization) ❌ Below target
- Three buffers: 2 GB + 2 GB + 2 GB = 6 GB total (80% utilization) ❌ Below target
- **Note**: With max_alloc = 2 GB, maximum possible is 6 GB (3 buffers × 2 GB), which is 80% of remaining memory. The algorithm will select 3 buffers as the best possible configuration.

### Scenario 2: Two Buffers Optimal
- Global memory: 8 GB
- Max alloc: 4 GB (each buffer limited to 4 GB)
- Remaining: 7.5 GB
- Target utilization: 7.425 GB (99% of 7.5 GB)
- Single buffer: 4 GB max (53% utilization) ❌ Below target
- Two buffers: 3.75 GB + 3.75 GB = 7.5 GB total (100% utilization) ✅ Meets target
- **Result**: 2 buffers (smallest number that meets 99% target)

### Scenario 3: Three Buffers Needed
- Global memory: 12 GB
- Max alloc: 2 GB (each buffer limited to 2 GB)
- Remaining: 11 GB
- Target utilization: 10.89 GB (99% of 11 GB)
- Single buffer: 2 GB max (18% utilization) ❌ Below target
- Two buffers: 2 GB + 2 GB = 4 GB total (36% utilization) ❌ Below target
- Three buffers: 2 GB + 2 GB + 2 GB = 6 GB total (54% utilization) ❌ Below target
- **Note**: With max_alloc = 2 GB, maximum possible is 6 GB (3 buffers × 2 GB), which is 54% of remaining memory. The algorithm will select 3 buffers as the best possible configuration, even though it doesn't meet the 99% target.

### Scenario 4: Single Buffer Sufficient
- Global memory: 8 GB
- Max alloc: 8 GB (no allocation limit)
- Remaining: 7.5 GB
- Target utilization: 7.425 GB (99% of 7.5 GB)
- Single buffer: 7.5 GB (100% utilization) ✅ Meets target
- **Result**: 1 buffer (meets 99% target, no need for multiple buffers)

### Key Insight

The algorithm's effectiveness depends on the ratio of `max_alloc` to `remaining_mem_size`:
- If `max_alloc` is large relative to remaining memory → 1 buffer may suffice
- If `max_alloc` is small relative to remaining memory → multiple buffers needed, but may still not reach 99% target
- Maximum possible utilization = `min(3 × max_alloc, remaining_mem_size)`

## Configuration

The feature is automatically enabled when:
1. `max_alloc < global_mem_size`
2. `remaining_mem_size > max_alloc`
3. Multiple buffers can improve utilization

No manual configuration required - the algorithm automatically selects the optimal configuration based on available memory and thread concurrency settings.

## Logging

The implementation logs:

### Memory Source Detection
- **AMD free memory**: Logs all 4 free memory values (KB) and selected value (bytes)
- **Standard global memory**: Logs total global memory size
- **Source indicator**: Final log shows which method was used (`CL_DEVICE_GLOBAL_FREE_MEMORY_AMD` or `CL_DEVICE_GLOBAL_MEM_SIZE`)

### Buffer Configuration
- **Calculation**: Number of buffers and groups per buffer
- **Memory utilization**: Total memory used and remaining unused memory (if any)
- **Utilization percentage**: Shows how effectively memory is being used

### Buffer Creation
- **Per buffer**: Size and memory type (host/device) for each buffer
- **Total buffers**: Number of buffers created and memory type used

### Error Handling
- **Memory overlap prevention**: Detailed error messages if total groups size exceeds remaining memory
  - Required memory breakdown (groups × bytes/group = total)
  - Available memory breakdown (remaining, global, other buffers)
  - Suggestion to reduce `thread_concurrency` or `lookup_gap`
- **Initialization failure**: YACMiner stops if memory validation fails

## Safety Features

### Memory Overlap Prevention

The implementation includes a critical safety check to prevent memory overlap:

1. **Pre-calculation validation**: Before buffer allocation, calculates total memory required for all groups
2. **Size comparison**: Compares `total_groups_size` against `remaining_mem_size`
3. **Early failure**: If memory requirements exceed available memory, YACMiner:
   - Logs detailed error messages showing:
     - Required memory (groups × bytes/group = total)
     - Available memory (remaining, global, other buffers)
   - Stops initialization immediately
   - Provides guidance to user (reduce `thread_concurrency` or `lookup_gap`)

This prevents:
- GPU crashes from memory access faults
- Memory corruption
- Incorrect mining results
- Silent failures

### Buffer Indexing Safety

The kernel implementation ensures correct buffer indexing:

1. **Buffer-relative indexing**: Each buffer uses its own `THREADS_PER_BUFFER_X` for indexing
2. **Offset calculation**: `relative_gid` is adjusted to be relative to the selected buffer
3. **ROMix parameter**: `xSIZE_override` ensures ROMix uses correct buffer size for indexing
4. **No out-of-bounds access**: Prevents memory access faults by using buffer-specific thread counts

## Future Enhancements

Potential improvements:
- Configurable utilization target (currently hardcoded at 99%)
- Support for more than 3 buffers if needed
- Dynamic buffer reallocation based on runtime conditions
- Performance metrics to validate buffer configuration choices
- Automatic thread_concurrency reduction with warning instead of failure (optional)

