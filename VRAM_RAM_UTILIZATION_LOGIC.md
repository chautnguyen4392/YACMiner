# VRAM & System RAM Utilization Logic

This document consolidates the design and implementation notes for distributing scrypt `padbuffer8` data across GPU VRAM and optional system RAM. It replaces the previous `MULTIPLE_PADBUFFER_LOGIC.md` and `SYSTEM_RAM_SUPPORT.md` files and reflects the logic currently implemented in `ocl.c` (lines 673-903).

## Goals
- Use as much working memory as possible while respecting `CL_DEVICE_MAX_MEM_ALLOC_SIZE`.
- Prefer fewer buffers when multiple configurations satisfy allocation limits.
- Keep buffer workloads balanced to minimise divergent execution paths.
- Prevent memory overlap between VRAM and system RAM allocations.

## Key Data & Terminology
- `each_group_size = 128 * ipt * wsize` bytes (one lookup group).
- `number_groups = thread_concurrency / wsize`.
- `max_alloc = cgpu->max_alloc`, `global_mem_size = cgpu->global_mem_size`.
- `padbuffer8[0..2]`: VRAM buffers (up to 3).
- `padbuffer8_RAM[0..1]`: System RAM buffers (up to 2) allocated with `CL_MEM_ALLOC_HOST_PTR`.

## Memory Source Detection
1. Attempt to use the AMD free memory extension (`CL_DEVICE_GLOBAL_FREE_MEMORY_AMD`) to obtain the largest free block in VRAM. If unavailable or it fails, fall back to `CL_DEVICE_GLOBAL_MEM_SIZE`.
2. When `--use-system-ram` is specified, compute available system memory per GPU:
   - Prefer `/proc/meminfo` (`MemAvailable`, `MemFree`, `MemTotal`), otherwise fall back to `sysinfo`.
   - Divide the detected available memory evenly across all GPUs (`clDevicesNum()`).
3. Log the source used and the detected amounts for traceability.

## Memory Accounting & Safety
- Reserve space for other GPU allocations before planning `padbuffer8`:
  - `CLbuffer0`, kernel output buffer, and optional split-kernel temporaries.
- `remaining_vram = global_mem_size - other_buffers_size` is an upper bound on VRAM usable by padbuffers.
- Total required bytes `total_groups_size = number_groups * each_group_size`.
- **Hard guard:** Abort initialization if `total_groups_size > remaining_vram + available_system_ram` (prevents overlap/corruption).

## VRAM Buffer Planning
1. `num_groups_for_vram = min(number_groups, remaining_vram / each_group_size)`.
2. `max_groups_per_buffer = max_alloc / each_group_size`. Abort if zero (group larger than alloc limit).
3. Determine if multiple buffers are required: `use_multiple_buffers = (remaining_vram > max_alloc)` and `num_groups_for_vram > 0`.
4. `required_buffers = ceil(num_groups_for_vram / max_groups_per_buffer)` capped at 3. Warn when capping.
5. Partition groups evenly:
   - For each buffer, assign `ceil(groups_remaining / buffers_remaining)` but clamp to `max_groups_per_buffer`.
   - Continue until all VRAM groups are assigned.
6. Record `clState->num_padbuffers` and `groups_per_buffer[i]`. Compute total VRAM bytes and log buffer counts and utilisation.

## System RAM Planning (Optional)
1. Compute leftover groups: `num_groups_for_ram = number_groups - sum(groups_per_buffer)`.
2. Abort if `num_groups_for_ram > available_system_ram / each_group_size` (insufficient RAM).
3. `max_groups_per_ram_buffer = max_alloc / each_group_size`. Warn (and skip RAM buffers) if zero.
4. `required_ram_buffers = ceil(num_groups_for_ram / max_groups_per_ram_buffer)` capped at 2 with a warning if capped.
5. Evenly distribute remaining groups using the same rounding strategy as VRAM, clamped per buffer.
6. Update `clState->num_padbuffers_RAM`, `groups_per_buffer_RAM[]`, total RAM bytes, and log utilisation. Warn when `--use-system-ram` is set but no groups remain.

## Final Validation
- Sum of groups assigned to all buffers must equal `number_groups`.
- Total allocated bytes must equal `total_groups_size`.
- Abort and log an error if either check fails.

## Kernel Integration
- All kernels accept the selected buffers: system RAM buffers first (prioritized), VRAM buffers second.
- Preprocessor defines communicate counts and sizes:
  - `NUM_PADBUFFERS`, `THREADS_PER_BUFFER_0/1/2`
  - `NUM_PADBUFFERS_RAM`, `THREADS_PER_BUFFER_RAM_0/1`
- Kernel dispatch logic:
  - Compute `group_id` via `get_group_id(0)` to avoid `global_work_offset` issues.
  - Determine `relative_gid`, select the corresponding buffer, adjust indices, and call `scrypt_ROMix` with buffer-specific `xSIZE_override`.

## Logging & Error Handling
- **Errors:** total requirement exceeds memory, a group cannot fit within `max_alloc`, system RAM insufficient, or final validation mismatches.
- **Warnings:** buffer requirements capped at implementation limits, system RAM option enabled with no leftover work, or per-buffer capacity of zero.
- **Info/Debug:** memory source diagnostics, buffer allocations, utilisation percentages for VRAM and RAM, and per-buffer sizes during creation/destruction.

## Future Improvements
- Configurable utilisation targets or dynamic buffer counts beyond current limits (3 VRAM / 2 RAM).
- Adaptive distribution across GPUs instead of equal RAM splitting.
- Runtime monitoring to trigger automatic workload adjustments instead of hard aborts.

This consolidated approach ensures buffers respect OpenCL constraints while maximising effective memory usage for scrypt mining workloads.

