#ifndef __OCL_H__
#define __OCL_H__

#include "config.h"

#include <stdbool.h>
#ifdef HAVE_OPENCL
#ifdef __APPLE_CC__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
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
	cl_mem padbuffer8;  // Uses CL_MEM_ALLOC_HOST_PTR for host-allocated pinned memory
	size_t padbufsize;
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
