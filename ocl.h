#ifndef __OCL_H__
#define __OCL_H__

#include "config.h"

#include <stdbool.h>
#ifdef HAVE_OPENCL
#ifdef __APPLE_CC__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>  // For AMD extensions like CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
#endif

#include "miner.h"

typedef struct {
	cl_context context;
	cl_kernel kernel;
	cl_command_queue commandQueue;
	cl_program program;
	cl_mem outputBuffer;
#ifdef USE_SCRYPT
	cl_mem CLbuffer0;
	cl_mem padbuffer8[3];  // Multiple buffers (up to 3) for better memory utilization
	size_t num_padbuffers;  // Number of padbuffer8 buffers (1-3)
	size_t groups_per_buffer[3];  // Number of groups per buffer
	cl_mem padbuffer8_RAM[2];  // System RAM buffers (up to 2) for additional memory
	size_t num_padbuffers_RAM;  // Number of padbuffer8_RAM buffers (0-2)
	size_t groups_per_buffer_RAM[2];  // Number of groups per buffer for system RAM
	void * cldata;
	// Split kernel support
	cl_kernel kernel_part1;
	cl_kernel kernel_part2;
	cl_kernel kernel_part3;
	cl_mem temp_X_buffer;   // Intermediate buffer for split kernels (Part 1 output)
	cl_mem temp_X2_buffer;  // Intermediate buffer for split kernels (Part 2 output)
	bool use_split_kernels;
#endif
	bool hasBitAlign;
	bool hasOpenCL11plus;
	bool hasOpenCL12plus;
	bool goffset;
	cl_uint vwidth;
	size_t max_work_size;
	size_t wsize;
	size_t compute_shaders;
	enum cl_kernels chosen_kernel;
} _clState;

extern char *file_contents(const char *filename, int *length);
extern int clDevicesNum(void);
extern _clState *initCl(unsigned int gpu, char *name, size_t nameSize);
#endif /* HAVE_OPENCL */
#endif /* __OCL_H__ */
