/*
 * Copyright 2011-2012 Con Kolivas
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include "config.h"
#ifdef HAVE_OPENCL

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>

#ifdef WIN32
	#include <winsock2.h>
#else
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <netdb.h>
#endif

#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

#include "findnonce.h"
#include "ocl.h"

int opt_platform_id = -1;

char *file_contents(const char *filename, int *length)
{
	char *fullpath = alloca(PATH_MAX);
	void *buffer;
	FILE *f;

	strcpy(fullpath, opt_kernel_path);
	strcat(fullpath, filename);

	/* Try in the optional kernel path or installed prefix first */
	f = fopen(fullpath, "rb");
	if (!f) {
		/* Then try from the path cgminer was called */
		strcpy(fullpath, cgminer_path);
		strcat(fullpath, filename);
		f = fopen(fullpath, "rb");
	}
	/* Finally try opening it directly */
	if (!f)
		f = fopen(filename, "rb");

	if (!f) {
		applog(LOG_ERR, "Unable to open %s or %s for reading", filename, fullpath);
		return NULL;
	}

	fseek(f, 0, SEEK_END);
	*length = ftell(f);
	fseek(f, 0, SEEK_SET);

	buffer = malloc(*length+1);
	*length = fread(buffer, 1, *length, f);
	fclose(f);
	((char*)buffer)[*length] = '\0';

	return (char*)buffer;
}

int clDevicesNum(void) {
	cl_int status;
	char pbuff[256];
	cl_uint numDevices;
	cl_uint numPlatforms;
	int most_devices = -1;
	cl_platform_id *platforms;
	cl_platform_id platform = NULL;
	unsigned int i, mdplatform = 0;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	/* If this fails, assume no GPUs. */
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: clGetPlatformsIDs failed (no OpenCL SDK installed?)", status);
		return -1;
	}

	if (numPlatforms == 0) {
		applog(LOG_ERR, "clGetPlatformsIDs returned no platforms (no OpenCL SDK installed?)");
		return -1;
	}

	platforms = (cl_platform_id *)alloca(numPlatforms*sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Getting Platform Ids. (clGetPlatformsIDs)", status);
		return -1;
	}

	for (i = 0; i < numPlatforms; i++) {
		if (opt_platform_id >= 0 && (int)i != opt_platform_id)
			continue;

		status = clGetPlatformInfo( platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuff), pbuff, NULL);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Getting Platform Info. (clGetPlatformInfo)", status);
			return -1;
		}
		platform = platforms[i];
		applog(LOG_INFO, "CL Platform %d vendor: %s", i, pbuff);
		status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pbuff), pbuff, NULL);
		if (status == CL_SUCCESS)
			applog(LOG_INFO, "CL Platform %d name: %s", i, pbuff);
		status = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(pbuff), pbuff, NULL);
		if (status == CL_SUCCESS)
			applog(LOG_INFO, "CL Platform %d version: %s", i, pbuff);
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
		if (status != CL_SUCCESS) {
			applog(LOG_INFO, "Error %d: Getting Device IDs (num)", status);
			continue;
		}
		applog(LOG_INFO, "Platform %d devices: %d", i, numDevices);
		if ((int)numDevices > most_devices) {
			most_devices = numDevices;
			mdplatform = i;
		}
		if (numDevices) {
			unsigned int j;
			cl_device_id *devices = (cl_device_id *)malloc(numDevices*sizeof(cl_device_id));

			clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
			for (j = 0; j < numDevices; j++) {
				clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(pbuff), pbuff, NULL);
				applog(LOG_INFO, "\t%i\t%s", j, pbuff);
			}
			free(devices);
		}
	}

	if (opt_platform_id < 0)
		opt_platform_id = mdplatform;;

	return most_devices;
}

static int advance(char **area, unsigned *remaining, const char *marker)
{
	char *find = memmem(*area, *remaining, marker, strlen(marker));

	if (!find) {
		applog(LOG_DEBUG, "Marker \"%s\" not found", marker);
		return 0;
	}
	*remaining -= find - *area;
	*area = find;
	return 1;
}

#define OP3_INST_BFE_UINT	4ULL
#define OP3_INST_BFE_INT	5ULL
#define OP3_INST_BFI_INT	6ULL
#define OP3_INST_BIT_ALIGN_INT	12ULL
#define OP3_INST_BYTE_ALIGN_INT	13ULL

void patch_opcodes(char *w, unsigned remaining)
{
	uint64_t *opcode = (uint64_t *)w;
	int patched = 0;
	int count_bfe_int = 0;
	int count_bfe_uint = 0;
	int count_byte_align = 0;
	while (42) {
		int clamp = (*opcode >> (32 + 31)) & 0x1;
		int dest_rel = (*opcode >> (32 + 28)) & 0x1;
		int alu_inst = (*opcode >> (32 + 13)) & 0x1f;
		int s2_neg = (*opcode >> (32 + 12)) & 0x1;
		int s2_rel = (*opcode >> (32 + 9)) & 0x1;
		int pred_sel = (*opcode >> 29) & 0x3;
		if (!clamp && !dest_rel && !s2_neg && !s2_rel && !pred_sel) {
			if (alu_inst == OP3_INST_BFE_INT) {
				count_bfe_int++;
			} else if (alu_inst == OP3_INST_BFE_UINT) {
				count_bfe_uint++;
			} else if (alu_inst == OP3_INST_BYTE_ALIGN_INT) {
				count_byte_align++;
				// patch this instruction to BFI_INT
				*opcode &= 0xfffc1fffffffffffULL;
				*opcode |= OP3_INST_BFI_INT << (32 + 13);
				patched++;
			}
		}
		if (remaining <= 8)
			break;
		opcode++;
		remaining -= 8;
	}
	applog(LOG_DEBUG, "Potential OP3 instructions identified: "
		"%i BFE_INT, %i BFE_UINT, %i BYTE_ALIGN",
		count_bfe_int, count_bfe_uint, count_byte_align);
	applog(LOG_DEBUG, "Patched a total of %i BFI_INT instructions", patched);
}

_clState *initCl(unsigned int gpu, char *name, size_t nameSize)
{
	_clState *clState = calloc(1, sizeof(_clState));
	bool patchbfi = false, prog_built = false;
	struct cgpu_info *cgpu = &gpus[gpu];
	cl_platform_id platform = NULL;
	char pbuff[256], vbuff[255];
	cl_platform_id* platforms;
	cl_uint preferred_vwidth;
	cl_device_id *devices;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_int status;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Getting Platforms. (clGetPlatformsIDs)", status);
		return NULL;
	}

	platforms = (cl_platform_id *)alloca(numPlatforms*sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Getting Platform Ids. (clGetPlatformsIDs)", status);
		return NULL;
	}

	if (opt_platform_id >= (int)numPlatforms) {
		applog(LOG_ERR, "Specified platform that does not exist");
		return NULL;
	}

	status = clGetPlatformInfo(platforms[opt_platform_id], CL_PLATFORM_VENDOR, sizeof(pbuff), pbuff, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Getting Platform Info. (clGetPlatformInfo)", status);
		return NULL;
	}
	platform = platforms[opt_platform_id];

	if (platform == NULL) {
		perror("NULL platform found!\n");
		return NULL;
	}

	applog(LOG_INFO, "CL Platform vendor: %s", pbuff);
	status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pbuff), pbuff, NULL);
	if (status == CL_SUCCESS)
		applog(LOG_INFO, "CL Platform name: %s", pbuff);
	status = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(vbuff), vbuff, NULL);
	if (status == CL_SUCCESS)
		applog(LOG_INFO, "CL Platform version: %s", vbuff);

	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Getting Device IDs (num)", status);
		return NULL;
	}

	if (numDevices > 0 ) {
		devices = (cl_device_id *)malloc(numDevices*sizeof(cl_device_id));

		/* Now, get the device list data */

		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Getting Device IDs (list)", status);
			return NULL;
		}

		applog(LOG_INFO, "List of devices:");

		unsigned int i;
		for (i = 0; i < numDevices; i++) {
			status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(pbuff), pbuff, NULL);
			if (status != CL_SUCCESS) {
				applog(LOG_ERR, "Error %d: Getting Device Info", status);
				return NULL;
			}

			applog(LOG_INFO, "\t%i\t%s", i, pbuff);
		}

		if (gpu < numDevices) {
			status = clGetDeviceInfo(devices[gpu], CL_DEVICE_NAME, sizeof(pbuff), pbuff, NULL);
			if (status != CL_SUCCESS) {
				applog(LOG_ERR, "Error %d: Getting Device Info", status);
				return NULL;
			}

			applog(LOG_INFO, "Selected %i: %s", gpu, pbuff);
			strncpy(name, pbuff, nameSize);
		} else {
			applog(LOG_ERR, "Invalid GPU %i", gpu);
			return NULL;
		}

	} else return NULL;

	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	clState->context = clCreateContextFromType(cps, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Context. (clCreateContextFromType)", status);
		return NULL;
	}

	/////////////////////////////////////////////////////////////////
	// Create an OpenCL command queue with profiling enabled
	/////////////////////////////////////////////////////////////////
	clState->commandQueue = clCreateCommandQueue(clState->context, devices[gpu],
						     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &status);
	if (status != CL_SUCCESS) /* Try again without OOE enable */
		clState->commandQueue = clCreateCommandQueue(clState->context, devices[gpu], CL_QUEUE_PROFILING_ENABLE, &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Command Queue. (clCreateCommandQueue)", status);
		return NULL;
	}

	/* Check for BFI INT support. Hopefully people don't mix devices with
	 * and without it! */
	char * extensions = malloc(1024);
	const char * camo = "cl_amd_media_ops";
	char *find;

	status = clGetDeviceInfo(devices[gpu], CL_DEVICE_EXTENSIONS, 1024, (void *)extensions, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_EXTENSIONS", status);
		return NULL;
	}
	applog(LOG_DEBUG, "Device Extensions: %s", extensions);
	find = strstr(extensions, camo);
	if (find)
		clState->hasBitAlign = true;
	applog(LOG_DEBUG, "Has Bit Align: %d", clState->hasBitAlign);
		
	/* Check for OpenCL >= 1.0 support, needed for global offset parameter usage. */
	char * devoclver = malloc(1024);
	const char * ocl10 = "OpenCL 1.0";
	const char * ocl11 = "OpenCL 1.1";

	status = clGetDeviceInfo(devices[gpu], CL_DEVICE_VERSION, 1024, (void *)devoclver, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_VERSION", status);
		return NULL;
	}
	applog(LOG_DEBUG, "Device Version: %s", devoclver);
	find = strstr(devoclver, ocl10);
	if (!find) {
		clState->hasOpenCL11plus = true;
		find = strstr(devoclver, ocl11);
		if (!find)
			clState->hasOpenCL12plus = true;
	}
	applog(LOG_DEBUG, "hasOpenCL11plus: %d", clState->hasOpenCL11plus);
	applog(LOG_DEBUG, "hasOpenCL12plus: %d", clState->hasOpenCL12plus);

	status = clGetDeviceInfo(devices[gpu], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), (void *)&preferred_vwidth, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT", status);
		return NULL;
	}
	applog(LOG_DEBUG, "Preferred vector width reported %d", preferred_vwidth);

	status = clGetDeviceInfo(devices[gpu], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), (void *)&clState->max_work_size, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_MAX_WORK_GROUP_SIZE", status);
		return NULL;
	}
	applog(LOG_DEBUG, "Max work group size reported %d", (int)(clState->max_work_size));

	size_t compute_units = 0;
	status = clGetDeviceInfo(devices[gpu], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), (void *)&compute_units, NULL);
	if (status != CL_SUCCESS) {
	applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_MAX_COMPUTE_UNITS", status);
	return NULL;
	}
	applog(LOG_DEBUG, "Max Compute units: %d", (int)(compute_units));

	// AMD architechture got 64 compute shaders per compute unit.
	// Source: http://www.amd.com/us/Documents/GCN_Architecture_whitepaper.pdf
	clState->compute_shaders = compute_units * 64;
	applog(LOG_DEBUG, "Max shaders calculated %d", (int)(clState->compute_shaders));

	status = clGetDeviceInfo(devices[gpu], CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(cl_ulong), (void *)&cgpu->max_alloc, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_MAX_MEM_ALLOC_SIZE", status);
		return NULL;
	}
	applog(LOG_DEBUG, "Max mem alloc size is %lu", (long unsigned int)(cgpu->max_alloc));

	// Try AMD free memory extension first (more accurate for allocation decisions)
	// CL_DEVICE_GLOBAL_FREE_MEMORY_AMD returns array of 4 size_t values (free memory in KB)
	// Use first element (largest free memory block), convert KB to bytes
	bool use_amd_free_mem = false;
	if (strstr(extensions, "cl_amd_device_attribute_query")) {
		size_t free_mem[4];
		status = clGetDeviceInfo(devices[gpu], CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, sizeof(free_mem), free_mem, NULL);
		if (status == CL_SUCCESS && free_mem[0] > 0) {
			cgpu->global_mem_size = (cl_ulong)free_mem[0] * 1024;
			use_amd_free_mem = true;
			applog(LOG_DEBUG, "AMD free memory (KB): [%zu, %zu, %zu, %zu], using %lu bytes", 
			       free_mem[0], free_mem[1], free_mem[2], free_mem[3], 
			       (long unsigned int)(cgpu->global_mem_size));
		}
	}

	// Fallback to standard global memory size if AMD extension not used
	if (!use_amd_free_mem) {
		status = clGetDeviceInfo(devices[gpu], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), (void *)&cgpu->global_mem_size, NULL);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_GLOBAL_MEM_SIZE", status);
			return NULL;
		}
	}

	applog(LOG_DEBUG, "Global memory size is %lu (from %s)", 
	       (long unsigned int)(cgpu->global_mem_size),
	       use_amd_free_mem ? "CL_DEVICE_GLOBAL_FREE_MEMORY_AMD" : "CL_DEVICE_GLOBAL_MEM_SIZE");

	/* Create binary filename based on parameters passed to opencl
	 * compiler to ensure we only load a binary that matches what would
	 * have otherwise created. The filename is:
	 * name + kernelname +/- g(offset) + v + vectors + w + work_size + l + sizeof(long) + .bin
	 * For scrypt the filename is:
	 * name + kernelname + g + lg + lookup_gap + tc + thread_concurrency + w + work_size + l + sizeof(long) + .bin
	 */
	char binaryfilename[255];
	char filename[255];
	char numbuf[16];

	if (cgpu->kernel == KL_NONE) {
		if (opt_scrypt) {
			if (opt_scrypt_chacha) {
				applog(LOG_INFO, "Selecting scrypt-chacha kernel");
				clState->chosen_kernel = KL_SCRYPT_CHACHA;
			} else if (opt_n_scrypt) {
				applog(LOG_INFO, "Selecting N-scrypt kernel");
				clState->chosen_kernel = KL_N_SCRYPT;
			} else {
				applog(LOG_INFO, "Selecting standard scrypt kernel");
				clState->chosen_kernel = KL_SCRYPT;
			}

		} else if (!strstr(name, "Tahiti") &&
			/* Detect all 2.6 SDKs not with Tahiti and use diablo kernel */
			(strstr(vbuff, "844.4") ||  // Linux 64 bit ATI 2.6 SDK
			 strstr(vbuff, "851.4") ||  // Windows 64 bit ""
			 strstr(vbuff, "831.4") ||
			 strstr(vbuff, "898.1") ||  // 12.2 driver SDK 
			 strstr(vbuff, "923.1") ||  // 12.4
			 strstr(vbuff, "938.2") ||  // SDK 2.7
			 strstr(vbuff, "1113.2"))) {// SDK 2.8
				applog(LOG_INFO, "Selecting diablo kernel");
				clState->chosen_kernel = KL_DIABLO;
		/* Detect all 7970s, older ATI and NVIDIA and use poclbm */
		} else if (strstr(name, "Tahiti") || !clState->hasBitAlign) {
			applog(LOG_INFO, "Selecting poclbm kernel");
			clState->chosen_kernel = KL_POCLBM;
		/* Use phatk for the rest R5xxx R6xxx */
		} else {
			applog(LOG_INFO, "Selecting phatk kernel");
			clState->chosen_kernel = KL_PHATK;
		}
		cgpu->kernel = clState->chosen_kernel;
	} else {
		clState->chosen_kernel = cgpu->kernel;
		if (clState->chosen_kernel == KL_PHATK &&
		    (strstr(vbuff, "844.4") || strstr(vbuff, "851.4") ||
		     strstr(vbuff, "831.4") || strstr(vbuff, "898.1") ||
		     strstr(vbuff, "923.1") || strstr(vbuff, "938.2") ||
		     strstr(vbuff, "1113.2"))) {
			applog(LOG_WARNING, "WARNING: You have selected the phatk kernel.");
			applog(LOG_WARNING, "You are running SDK 2.6+ which performs poorly with this kernel.");
			applog(LOG_WARNING, "Downgrade your SDK and delete any .bin files before starting again.");
			applog(LOG_WARNING, "Or allow cgminer to automatically choose a more suitable kernel.");
		}
	}

	/* For some reason 2 vectors is still better even if the card says
	 * otherwise, and many cards lie about their max so use 256 as max
	 * unless explicitly set on the command line. Tahiti prefers 1 */
	if (strstr(name, "Tahiti"))
		preferred_vwidth = 1;
	else if (preferred_vwidth > 2)
		preferred_vwidth = 2;

	switch (clState->chosen_kernel) {
		case KL_POCLBM:
			strcpy(filename, POCLBM_KERNNAME".cl");
			strcpy(binaryfilename, POCLBM_KERNNAME);
			break;
		case KL_PHATK:
			strcpy(filename, PHATK_KERNNAME".cl");
			strcpy(binaryfilename, PHATK_KERNNAME);
			break;
		case KL_DIAKGCN:
			strcpy(filename, DIAKGCN_KERNNAME".cl");
			strcpy(binaryfilename, DIAKGCN_KERNNAME);
			break;
		case KL_SCRYPT:
			strcpy(filename, SCRYPT_KERNNAME".cl");
			strcpy(binaryfilename, SCRYPT_KERNNAME);
			/* Scrypt only supports vector 1 */
			cgpu->vwidth = 1;
			break;
		case KL_N_SCRYPT:
			strcpy(filename, N_SCRYPT_KERNNAME".cl");
			strcpy(binaryfilename, N_SCRYPT_KERNNAME);
			/* Scrypt only supports vector 1 */
			cgpu->vwidth = 1;
			break;
		case KL_SCRYPT_CHACHA:
			strcpy(filename, SCRYPT_CHACHA_KERNNAME".cl");
			strcpy(binaryfilename, SCRYPT_CHACHA_KERNNAME);
			/* Scrypt only supports vector 1 */
			cgpu->vwidth = 1;
			break;
		case KL_NONE: /* Shouldn't happen */
		case KL_DIABLO:
			strcpy(filename, DIABLO_KERNNAME".cl");
			strcpy(binaryfilename, DIABLO_KERNNAME);
			break;
	}

	if (cgpu->vwidth)
		clState->vwidth = cgpu->vwidth;
	else {
		clState->vwidth = preferred_vwidth;
		cgpu->vwidth = preferred_vwidth;
	}

	if (((clState->chosen_kernel == KL_POCLBM || clState->chosen_kernel == KL_DIABLO || clState->chosen_kernel == KL_DIAKGCN) &&
		clState->vwidth == 1 && clState->hasOpenCL11plus) || opt_scrypt)
			clState->goffset = true;

	if (cgpu->work_size && cgpu->work_size <= clState->max_work_size)
		clState->wsize = cgpu->work_size;
	else if (opt_scrypt_chacha)
		clState->wsize = 12;
	else if (opt_scrypt)
		clState->wsize = 256;
	else if (strstr(name, "Tahiti"))
		clState->wsize = 64;
	else
		clState->wsize = (clState->max_work_size <= 256 ? clState->max_work_size : 256) / clState->vwidth;
	cgpu->work_size = clState->wsize;
	applog(LOG_DEBUG, "Work size: %d", (int)(clState->wsize));

#ifdef USE_SCRYPT
	if (opt_scrypt) {
		if (!cgpu->opt_lg) {
			applog(LOG_NOTICE, "GPU %d: selecting lookup gap of 32", gpu);
			cgpu->lookup_gap = 32;
		} else
			cgpu->lookup_gap = cgpu->opt_lg;

		unsigned long bsize;
		if (opt_scrypt_chacha && opt_fixed_nfactor > 0)
			bsize = 1 << (opt_fixed_nfactor + 1);
		else if (opt_n_scrypt)
			bsize = 2048;
		else
			bsize = 1024;
		size_t ipt = (bsize / cgpu->lookup_gap + (bsize % cgpu->lookup_gap > 0));

		// if we do not have TC and we do not have BS, then calculate some conservative numbers
		if ((!cgpu->opt_tc) && (!cgpu->buffer_size)) {
			unsigned long base_alloc;

			// default to 100% of the available memory and find the closest MB value divisible by 8
			base_alloc = (unsigned long)(cgpu->max_alloc * 100 / 100 / 1024 / 1024 / 8) * 8 * 1024 * 1024 / cgpu->threads;
			// base_alloc is now the number of bytes to allocate.  
			// 2 threads of 336 MB did not fit into dedicated VRAM while 1 thread of 772MB did.  334 MB each did
			// to be safe, reduce by 2MB per thread beyond the first

			base_alloc -= (cgpu->threads - 1) * 2 * 1024 * 1024;

			cgpu->thread_concurrency = base_alloc / 128 / ipt;
			cgpu->buffer_size = base_alloc / 1024 / 1024;
			applog(LOG_DEBUG,"88%% Max Allocation: %lu",base_alloc);
			applog(LOG_NOTICE, "GPU %d: selecting buffer_size of %zu", gpu, cgpu->buffer_size);
		} else
			cgpu->thread_concurrency = cgpu->opt_tc;

		if (cgpu->buffer_size) {
			// use the buffer-size to overwrite the thread-concurrency
			cgpu->thread_concurrency = (int)((cgpu->buffer_size * 1024 * 1024) / ipt / 128);
			applog(LOG_DEBUG, "GPU %d: setting thread_concurrency to %lu based on buffer size %lu and lookup gap %d", gpu, (unsigned long)(cgpu->thread_concurrency),(unsigned long)(cgpu->buffer_size),(int)(cgpu->lookup_gap));
		}

		// Calculate optimal buffer configuration for multiple padbuffer8 buffers
		// Initialize padbuffer8 array
		clState->num_padbuffers = 0;
		for (int i = 0; i < 3; i++) {
			clState->padbuffer8[i] = NULL;
			clState->groups_per_buffer[i] = 0;
		}

		size_t each_item_size = 128 * ipt;
		size_t each_group_size = each_item_size * clState->wsize;
		size_t number_groups = cgpu->thread_concurrency / clState->wsize;

		// Calculate remaining memory after other buffers (conservative estimate)
		size_t CLbuffer0_size = 128;
		size_t outputBuffer_size = SCRYPT_BUFFERSIZE;
		// Estimate temp buffers (will be created later if split kernels enabled)
		size_t temp_X_size = cgpu->thread_concurrency * 8 * sizeof(cl_uint4);
		size_t temp_X2_size = temp_X_size;
		size_t other_buffers_size = CLbuffer0_size + outputBuffer_size + temp_X_size + temp_X2_size;
		
		cl_ulong remaining_mem_size = 0;
		bool use_multiple_buffers = false;
		
		if (cgpu->max_alloc < cgpu->global_mem_size) {
			if (cgpu->global_mem_size > other_buffers_size) {
				remaining_mem_size = cgpu->global_mem_size - other_buffers_size;
				if (remaining_mem_size > cgpu->max_alloc) {
					use_multiple_buffers = true;
				}
			}
		}

		// Ensure total size for all groups fits within remaining_mem_size to prevent memory overlap
		cl_ulong total_groups_size = (cl_ulong)number_groups * each_group_size;
		if (remaining_mem_size > 0 && total_groups_size > remaining_mem_size) {
			applog(LOG_ERR, "GPU %d: Total groups size (%lu bytes) exceeds remaining memory (%lu bytes). "
			       "This would cause memory overlap. Please reduce thread_concurrency or lookup_gap.", 
			       gpu, (long unsigned int)total_groups_size, (long unsigned int)remaining_mem_size);
			applog(LOG_ERR, "GPU %d: Required: %zu groups * %zu bytes/group = %lu bytes", 
			       gpu, number_groups, each_group_size, (long unsigned int)total_groups_size);
			applog(LOG_ERR, "GPU %d: Available: %lu bytes (global_mem: %lu, other_buffers: %zu)", 
			       gpu, (long unsigned int)remaining_mem_size, 
			       (long unsigned int)cgpu->global_mem_size, other_buffers_size);
			return NULL;
		}

		size_t buffer_size = 0;
		size_t optimal_num_buffers = 1;
		size_t optimal_groups_per_buffer[3] = {0, 0, 0};
		cl_ulong best_utilization = 0;  // Track best memory utilization

		if (use_multiple_buffers && number_groups > 0) {
			// Try configurations starting from 1 buffer, find smallest number that maximizes utilization
			// Target: use at least 99% of remaining memory
			cl_ulong target_utilization = remaining_mem_size * 99 / 100;
			
			// Try 1 buffer first
			size_t single_buffer_size = each_group_size * number_groups;
			if (single_buffer_size > cgpu->max_alloc) {
				single_buffer_size = cgpu->max_alloc;
			}
			cl_ulong single_buffer_utilization = single_buffer_size;
			if (single_buffer_size <= cgpu->max_alloc && single_buffer_utilization >= target_utilization) {
				optimal_num_buffers = 1;
				buffer_size = single_buffer_size;
				optimal_groups_per_buffer[0] = buffer_size / each_group_size;
				best_utilization = single_buffer_utilization;
				goto found_optimal_config;
			}
			// Track best so far even if below target
			if (single_buffer_utilization > best_utilization) {
				optimal_num_buffers = 1;
				buffer_size = single_buffer_size;
				optimal_groups_per_buffer[0] = buffer_size / each_group_size;
				best_utilization = single_buffer_utilization;
			}
			
			// Try 2 buffers
			size_t best_2buf_config[2] = {0, 0};
			cl_ulong best_2buf_utilization = 0;
			for (size_t groups_buf1 = number_groups / 2; groups_buf1 > 0; groups_buf1--) {
				size_t groups_buf2 = number_groups - groups_buf1;
				size_t buf1_size = each_group_size * groups_buf1;
				size_t buf2_size = each_group_size * groups_buf2;
				if (buf1_size <= cgpu->max_alloc && buf2_size <= cgpu->max_alloc) {
					cl_ulong total_size = buf1_size + buf2_size;
					if (total_size <= remaining_mem_size && total_size > best_2buf_utilization) {
						best_2buf_config[0] = groups_buf1;
						best_2buf_config[1] = groups_buf2;
						best_2buf_utilization = total_size;
					}
				}
			}
			if (best_2buf_utilization > best_utilization) {
				optimal_num_buffers = 2;
				optimal_groups_per_buffer[0] = best_2buf_config[0];
				optimal_groups_per_buffer[1] = best_2buf_config[1];
				buffer_size = (each_group_size * best_2buf_config[0] > each_group_size * best_2buf_config[1]) ?
				              (each_group_size * best_2buf_config[0]) : (each_group_size * best_2buf_config[1]);
				best_utilization = best_2buf_utilization;
				// If this meets target, use it (smallest number that works)
				if (best_2buf_utilization >= target_utilization) {
					goto found_optimal_config;
				}
			}
			
			// Try 3 buffers only if 2 buffers didn't meet target
			if (best_utilization < target_utilization) {
				size_t best_3buf_config[3] = {0, 0, 0};
				cl_ulong best_3buf_utilization = 0;
				for (size_t groups_buf1 = number_groups / 3; groups_buf1 > 0; groups_buf1--) {
					for (size_t groups_buf2 = (number_groups - groups_buf1) / 2; groups_buf2 > 0; groups_buf2--) {
						size_t groups_buf3 = number_groups - groups_buf1 - groups_buf2;
						size_t buf1_size = each_group_size * groups_buf1;
						size_t buf2_size = each_group_size * groups_buf2;
						size_t buf3_size = each_group_size * groups_buf3;
						size_t max_buf_size = (buf1_size > buf2_size) ? 
							((buf1_size > buf3_size) ? buf1_size : buf3_size) :
							((buf2_size > buf3_size) ? buf2_size : buf3_size);
						if (max_buf_size <= cgpu->max_alloc) {
							cl_ulong total_size = buf1_size + buf2_size + buf3_size;
							if (total_size <= remaining_mem_size && total_size > best_3buf_utilization) {
								best_3buf_config[0] = groups_buf1;
								best_3buf_config[1] = groups_buf2;
								best_3buf_config[2] = groups_buf3;
								best_3buf_utilization = total_size;
							}
						}
					}
				}
				if (best_3buf_utilization > best_utilization) {
					optimal_num_buffers = 3;
					optimal_groups_per_buffer[0] = best_3buf_config[0];
					optimal_groups_per_buffer[1] = best_3buf_config[1];
					optimal_groups_per_buffer[2] = best_3buf_config[2];
					size_t buf1_size = each_group_size * best_3buf_config[0];
					size_t buf2_size = each_group_size * best_3buf_config[1];
					size_t buf3_size = each_group_size * best_3buf_config[2];
					buffer_size = (buf1_size > buf2_size) ? 
						((buf1_size > buf3_size) ? buf1_size : buf3_size) :
						((buf2_size > buf3_size) ? buf2_size : buf3_size);
					best_utilization = best_3buf_utilization;
				}
			}
		} else {
			// Fallback to single buffer if multiple buffers not applicable
			buffer_size = each_group_size * number_groups;
			if (buffer_size > cgpu->max_alloc) {
				buffer_size = cgpu->max_alloc;
				// Recalculate number_groups based on max_alloc
				number_groups = buffer_size / each_group_size;
			}
			optimal_groups_per_buffer[0] = number_groups;
		}
		
found_optimal_config:
		clState->num_padbuffers = optimal_num_buffers;
		for (int i = 0; i < optimal_num_buffers; i++) {
			clState->groups_per_buffer[i] = optimal_groups_per_buffer[i];
		}
		clState->padbufsize = buffer_size;

		// Calculate total memory used by padbuffer8 buffers
		cl_ulong total_padbuffer_mem = 0;
		for (int i = 0; i < optimal_num_buffers; i++) {
			total_padbuffer_mem += each_group_size * optimal_groups_per_buffer[i];
		}
		
		// Calculate remaining unused memory
		cl_ulong unused_mem = 0;
		if (use_multiple_buffers && remaining_mem_size > 0) {
			if (total_padbuffer_mem < remaining_mem_size) {
				unused_mem = remaining_mem_size - total_padbuffer_mem;
			}
		}

		applog(LOG_DEBUG, "GPU %d: Calculated buffer config: %zu buffers, groups per buffer: [%zu, %zu, %zu]",
		       gpu, clState->num_padbuffers,
		       clState->groups_per_buffer[0], clState->groups_per_buffer[1], clState->groups_per_buffer[2]);
		
		if (unused_mem > 0) {
			applog(LOG_INFO, "GPU %d: padbuffer8 buffers use %lu MB, %lu MB remaining unused (%.1f%% utilization)",
			       gpu, (unsigned long)(total_padbuffer_mem / (1024 * 1024)),
			       (unsigned long)(unused_mem / (1024 * 1024)),
			       (double)(total_padbuffer_mem * 100.0 / remaining_mem_size));
		} else if (use_multiple_buffers && remaining_mem_size > 0) {
			applog(LOG_INFO, "GPU %d: padbuffer8 buffers use %lu MB (100%% utilization)",
			       gpu, (unsigned long)(total_padbuffer_mem / (1024 * 1024)));
		}
	}
#endif

	FILE *binaryfile;
	size_t *binary_sizes;
	char **binaries;
	int pl;
	char *source = file_contents(filename, &pl);
	applog(LOG_DEBUG, "filename: %s", filename);
	size_t sourceSize[] = {(size_t)pl};
	cl_uint slot, cpnd;

	slot = cpnd = 0;

	if (!source)
		return NULL;

	binary_sizes = calloc(sizeof(size_t) * MAX_GPUDEVICES * 4, 1);
	if (unlikely(!binary_sizes)) {
		applog(LOG_ERR, "Unable to calloc binary_sizes");
		return NULL;
	}
	binaries = calloc(sizeof(char *) * MAX_GPUDEVICES * 4, 1);
	if (unlikely(!binaries)) {
		applog(LOG_ERR, "Unable to calloc binaries");
		return NULL;
	}

	strcat(binaryfilename, name);
	if (clState->goffset)
		strcat(binaryfilename, "g");
	if (opt_scrypt) {
#ifdef USE_SCRYPT
		sprintf(numbuf, "lg%utc%u", cgpu->lookup_gap, (unsigned int)cgpu->thread_concurrency);
		strcat(binaryfilename, numbuf);
#endif
	} else {
		sprintf(numbuf, "v%d", clState->vwidth);
		strcat(binaryfilename, numbuf);
	}
	sprintf(numbuf, "w%d", (int)clState->wsize);
	strcat(binaryfilename, numbuf);
	sprintf(numbuf, "l%d", (int)sizeof(long));
	strcat(binaryfilename, numbuf);
	strcat(binaryfilename, ".bin");

	binaryfile = fopen(binaryfilename, "rb");
	applog(LOG_DEBUG, "binaryfilename: %s", binaryfilename);
	if (!binaryfile) {
		applog(LOG_DEBUG, "No binary found, generating from source");
	} else {
		struct stat binary_stat;

		if (unlikely(stat(binaryfilename, &binary_stat))) {
			applog(LOG_DEBUG, "Unable to stat binary, generating from source");
			fclose(binaryfile);
			goto build;
		}
		if (!binary_stat.st_size)
			goto build;

		binary_sizes[slot] = binary_stat.st_size;
		binaries[slot] = (char *)calloc(binary_sizes[slot], 1);
		if (unlikely(!binaries[slot])) {
			applog(LOG_ERR, "Unable to calloc binaries");
			fclose(binaryfile);
			return NULL;
		}

		if (fread(binaries[slot], 1, binary_sizes[slot], binaryfile) != binary_sizes[slot]) {
			applog(LOG_ERR, "Unable to fread binaries");
			fclose(binaryfile);
			free(binaries[slot]);
			goto build;
		}

		clState->program = clCreateProgramWithBinary(clState->context, 1, &devices[gpu], &binary_sizes[slot], (const unsigned char **)binaries, &status, NULL);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Loading Binary into cl_program (clCreateProgramWithBinary)", status);
			fclose(binaryfile);
			free(binaries[slot]);
			goto build;
		}

		fclose(binaryfile);
		applog(LOG_DEBUG, "Loaded binary image %s", binaryfilename);

		goto built;
	}

	/////////////////////////////////////////////////////////////////
	// Load CL file, build CL program object, create CL kernel object
	/////////////////////////////////////////////////////////////////

build:
	clState->program = clCreateProgramWithSource(clState->context, 1, (const char **)&source, sourceSize, &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Loading Binary into cl_program (clCreateProgramWithSource)", status);
		return NULL;
	}

	/* create a cl program executable for all the devices specified */
	char *CompilerOptions = calloc(1, 512);  // Increased size for multiple buffer defines

#ifdef USE_SCRYPT
	if (opt_scrypt)
	{
		// Calculate threads per buffer for ROMix indexing
		size_t threads_per_buffer[3];
		threads_per_buffer[0] = clState->groups_per_buffer[0] * clState->wsize;
		threads_per_buffer[1] = clState->groups_per_buffer[1] * clState->wsize;
		threads_per_buffer[2] = clState->groups_per_buffer[2] * clState->wsize;
		
		sprintf(CompilerOptions, "-D LOOKUP_GAP=%d -D CONCURRENT_THREADS=%d -D WORKSIZE=%d -D NUM_PADBUFFERS=%zu -D THREADS_PER_BUFFER_0=%zu -D THREADS_PER_BUFFER_1=%zu -D THREADS_PER_BUFFER_2=%zu",
			cgpu->lookup_gap, (unsigned int)cgpu->thread_concurrency, (int)clState->wsize,
			clState->num_padbuffers,
			threads_per_buffer[0], threads_per_buffer[1], threads_per_buffer[2]);
	}
	else
#endif
	{
		sprintf(CompilerOptions, "-D WORKSIZE=%d -D VECTORS%d -D WORKVEC=%d",
			(int)clState->wsize, clState->vwidth, (int)clState->wsize * clState->vwidth);
	}
	applog(LOG_DEBUG, "Setting worksize to %d", (int)(clState->wsize));
	if (clState->vwidth > 1)
		applog(LOG_DEBUG, "Patched source to suit %d vectors", clState->vwidth);

	if (clState->hasBitAlign) {
		strcat(CompilerOptions, " -D BITALIGN");
		applog(LOG_DEBUG, "cl_amd_media_ops found, setting BITALIGN");
		if (!clState->hasOpenCL12plus &&
		    (strstr(name, "Cedar") ||
		     strstr(name, "Redwood") ||
		     strstr(name, "Juniper") ||
		     strstr(name, "Cypress" ) ||
		     strstr(name, "Hemlock" ) ||
		     strstr(name, "Caicos" ) ||
		     strstr(name, "Turks" ) ||
		     strstr(name, "Barts" ) ||
		     strstr(name, "Cayman" ) ||
		     strstr(name, "Antilles" ) ||
		     strstr(name, "Wrestler" ) ||
		     strstr(name, "Zacate" ) ||
		     strstr(name, "WinterPark" )))
			patchbfi = true;
	} else
		applog(LOG_DEBUG, "cl_amd_media_ops not found, will not set BITALIGN");

	if (patchbfi) {
		strcat(CompilerOptions, " -D BFI_INT");
		applog(LOG_DEBUG, "BFI_INT patch requiring device found, patched source with BFI_INT");
	} else
		applog(LOG_DEBUG, "BFI_INT patch requiring device not found, will not BFI_INT patch");

	if (clState->goffset)
		strcat(CompilerOptions, " -D GOFFSET");

	if (!clState->hasOpenCL11plus)
		strcat(CompilerOptions, " -D OCL1");

	applog(LOG_DEBUG, "CompilerOptions: %s", CompilerOptions);
	status = clBuildProgram(clState->program, 1, &devices[gpu], CompilerOptions , NULL, NULL);
	free(CompilerOptions);

	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Building Program (clBuildProgram)", status);
		size_t logSize;
		status = clGetProgramBuildInfo(clState->program, devices[gpu], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

		char *log = malloc(logSize);
		status = clGetProgramBuildInfo(clState->program, devices[gpu], CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
		applog(LOG_ERR, "%s", log);
		return NULL;
	} else {
		applog(LOG_DEBUG, "Success: Building Program (clBuildProgram)");
	}

	prog_built = true;

#ifdef __APPLE__
	/* OSX OpenCL breaks reading off binaries with >1 GPU so always build
	 * from source. */
	goto built;
#endif

	status = clGetProgramInfo(clState->program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &cpnd, NULL);
	if (unlikely(status != CL_SUCCESS)) {
		applog(LOG_ERR, "Error %d: Getting program info CL_PROGRAM_NUM_DEVICES. (clGetProgramInfo)", status);
		return NULL;
	}

	status = clGetProgramInfo(clState->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*cpnd, binary_sizes, NULL);
	if (unlikely(status != CL_SUCCESS)) {
		applog(LOG_ERR, "Error %d: Getting program info CL_PROGRAM_BINARY_SIZES. (clGetProgramInfo)", status);
		return NULL;
	}

	/* The actual compiled binary ends up in a RANDOM slot! Grr, so we have
	 * to iterate over all the binary slots and find where the real program
	 * is. What the heck is this!? */
	for (slot = 0; slot < cpnd; slot++)
		if (binary_sizes[slot])
			break;

	/* copy over all of the generated binaries. */
	applog(LOG_DEBUG, "Binary size for gpu %d found in binary slot %d: %d", gpu, slot, (int)(binary_sizes[slot]));
	if (!binary_sizes[slot]) {
		applog(LOG_ERR, "OpenCL compiler generated a zero sized binary, FAIL!");
		return NULL;
	}
	binaries[slot] = calloc(sizeof(char) * binary_sizes[slot], 1);
	status = clGetProgramInfo(clState->program, CL_PROGRAM_BINARIES, sizeof(char *) * cpnd, binaries, NULL );
	if (unlikely(status != CL_SUCCESS)) {
		applog(LOG_ERR, "Error %d: Getting program info. CL_PROGRAM_BINARIES (clGetProgramInfo)", status);
		return NULL;
	}

	/* Patch the kernel if the hardware supports BFI_INT but it needs to
	 * be hacked in */
	if (patchbfi) {
		unsigned remaining = binary_sizes[slot];
		char *w = binaries[slot];
		unsigned int start, length;

		/* Find 2nd incidence of .text, and copy the program's
		* position and length at a fixed offset from that. Then go
		* back and find the 2nd incidence of \x7ELF (rewind by one
		* from ELF) and then patch the opcocdes */
		if (!advance(&w, &remaining, ".text"))
			goto build;
		w++; remaining--;
		if (!advance(&w, &remaining, ".text")) {
			/* 32 bit builds only one ELF */
			w--; remaining++;
		}
		memcpy(&start, w + 285, 4);
		memcpy(&length, w + 289, 4);
		w = binaries[slot]; remaining = binary_sizes[slot];
		if (!advance(&w, &remaining, "ELF"))
			goto build;
		w++; remaining--;
		if (!advance(&w, &remaining, "ELF")) {
			/* 32 bit builds only one ELF */
			w--; remaining++;
		}
		w--; remaining++;
		w += start; remaining -= start;
		applog(LOG_DEBUG, "At %p (%u rem. bytes), to begin patching", w, remaining);
		patch_opcodes(w, length);

		status = clReleaseProgram(clState->program);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Releasing program. (clReleaseProgram)", status);
			return NULL;
		}

		clState->program = clCreateProgramWithBinary(clState->context, 1, &devices[gpu], &binary_sizes[slot], (const unsigned char **)&binaries[slot], &status, NULL);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Loading Binary into cl_program (clCreateProgramWithBinary)", status);
			return NULL;
		}

		/* Program needs to be rebuilt */
		prog_built = false;
	}

	free(source);

	/* Save the binary to be loaded next time */
	binaryfile = fopen(binaryfilename, "wb");
	if (!binaryfile) {
		/* Not a fatal problem, just means we build it again next time */
		applog(LOG_DEBUG, "Unable to create file %s", binaryfilename);
	} else {
		if (unlikely(fwrite(binaries[slot], 1, binary_sizes[slot], binaryfile) != binary_sizes[slot])) {
			applog(LOG_ERR, "Unable to fwrite to binaryfile");
			return NULL;
		}
		fclose(binaryfile);
	}
built:
	if (binaries[slot])
		free(binaries[slot]);
	free(binaries);
	free(binary_sizes);

	applog(LOG_INFO, "Initialising kernel %s with%s bitalign, %d vectors and worksize %d",
	       filename, clState->hasBitAlign ? "" : "out", clState->vwidth, (int)(clState->wsize));

	if (!prog_built) {
		/* create a cl program executable for all the devices specified */
		status = clBuildProgram(clState->program, 1, &devices[gpu], NULL, NULL, NULL);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Building Program (clBuildProgram)", status);
			size_t logSize;
			status = clGetProgramBuildInfo(clState->program, devices[gpu], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

			char *log = malloc(logSize);
			status = clGetProgramBuildInfo(clState->program, devices[gpu], CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
			applog(LOG_ERR, "%s", log);
			return NULL;
		}
	}

	/* get a kernel object handle for a kernel with the given name */
	const char *kernel_name = (opt_scrypt_chacha_84) ? "search84" : "search";
	
	// Determine if we should use split kernels
	clState->use_split_kernels = false;
#ifdef USE_SCRYPT
	if (opt_scrypt_split_kernels && opt_scrypt_chacha_84) {
		clState->use_split_kernels = true;
		applog(LOG_INFO, "Using split kernel mode for reduced register pressure");
	}
#endif
	
	if (clState->use_split_kernels) {
		// Create three separate kernels for split execution
		clState->kernel_part1 = clCreateKernel(clState->program, "search84_part1", &status);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Creating Kernel Part 1. (clCreateKernel)", status);
			return NULL;
		}
		
		clState->kernel_part2 = clCreateKernel(clState->program, "search84_part2", &status);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Creating Kernel Part 2. (clCreateKernel)", status);
			clReleaseKernel(clState->kernel_part1);
			return NULL;
		}
		
		clState->kernel_part3 = clCreateKernel(clState->program, "search84_part3", &status);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Creating Kernel Part 3. (clCreateKernel)", status);
			clReleaseKernel(clState->kernel_part1);
			clReleaseKernel(clState->kernel_part2);
			return NULL;
		}
		
		// Also create monolithic kernel as fallback (stored in clState->kernel)
		clState->kernel = clCreateKernel(clState->program, kernel_name, &status);
		if (status != CL_SUCCESS) {
			applog(LOG_WARNING, "Could not create fallback kernel, continuing with split kernels only");
			clState->kernel = NULL;
		}
		
		applog(LOG_INFO, "Split kernels created successfully (Part 1, 2, 3)");
	} else {
		// Create single monolithic kernel
		clState->kernel = clCreateKernel(clState->program, kernel_name, &status);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Creating Kernel from program. (clCreateKernel)", status);
			return NULL;
		}
	}

#ifdef USE_SCRYPT
	if (opt_scrypt) {
		// Buffer configuration was already calculated earlier, now create the buffers
		unsigned long bsize;
		if (opt_scrypt_chacha && opt_fixed_nfactor > 0)
			bsize = 1 << (opt_fixed_nfactor + 1);
		else if (opt_n_scrypt)
			bsize = 2048;
		else
			bsize = 1024;

		size_t ipt = (bsize / cgpu->lookup_gap + (bsize % cgpu->lookup_gap > 0));
		size_t each_item_size = 128 * ipt;
		size_t each_group_size = each_item_size * clState->wsize;

		applog(LOG_INFO, "GPU %d: Creating %zu padbuffer8 buffer(s), groups per buffer: [%zu, %zu, %zu]",
		       gpu, clState->num_padbuffers,
		       clState->groups_per_buffer[0], clState->groups_per_buffer[1], clState->groups_per_buffer[2]);

		/* Create buffers - use host memory if option is enabled, otherwise use device memory */
		cl_mem_flags flags = CL_MEM_READ_WRITE;
		bool use_host_memory = false;
		
		if (opt_padbuffer_host_memory) {
			flags |= CL_MEM_ALLOC_HOST_PTR;
			use_host_memory = true;
		}
		
		// Create all padbuffer8 buffers
		for (size_t i = 0; i < clState->num_padbuffers; i++) {
			size_t buf_size = each_group_size * clState->groups_per_buffer[i];
			clState->padbuffer8[i] = clCreateBuffer(clState->context, flags, buf_size, NULL, &status);
			
			/* Fallback to device memory if host allocation fails */
			if ((status != CL_SUCCESS || !clState->padbuffer8[i]) && use_host_memory) {
				applog(LOG_WARNING, "Host-allocated memory failed for buffer %zu (error %d), falling back to device memory", i, status);
				use_host_memory = false;
				clState->padbuffer8[i] = clCreateBuffer(clState->context, CL_MEM_READ_WRITE, 
				                                    buf_size, NULL, &status);
			}
			
			if (status != CL_SUCCESS || !clState->padbuffer8[i]) {
				applog(LOG_ERR, "Error %d: clCreateBuffer (padbuffer8[%zu]) failed, size: %zu bytes", status, i, (unsigned long)buf_size);
				// Release already created buffers
				for (size_t j = 0; j < i; j++) {
					if (clState->padbuffer8[j]) {
						clReleaseMemObject(clState->padbuffer8[j]);
						clState->padbuffer8[j] = NULL;
					}
				}
				return NULL;
			}
			applog(LOG_DEBUG, "Created padbuffer8[%zu]: %zu bytes (%zu MB)", i, 
			       (unsigned long)buf_size, (unsigned long)(buf_size / (1024 * 1024)));
		}
		
		applog(LOG_INFO, "Created %zu padbuffer8 buffer(s) using %s", 
		       clState->num_padbuffers,
		       use_host_memory ? "OpenCL-allocated host (pinned) memory" : "device memory");

		clState->CLbuffer0 = clCreateBuffer(clState->context, CL_MEM_READ_ONLY, 128, NULL, &status);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: clCreateBuffer (CLbuffer0)", status);
			return NULL;
		}
		clState->outputBuffer = clCreateBuffer(clState->context, CL_MEM_WRITE_ONLY, SCRYPT_BUFFERSIZE, NULL, &status);
		
		// Create temp_X and temp_X2 buffers for split kernels if enabled
		if (clState->use_split_kernels) {
			// temp_X and temp_X2 need to hold 8 uint4 values per thread
			// Size = thread_concurrency * 8 * sizeof(cl_uint4)
			size_t temp_X_size = cgpu->thread_concurrency * 8 * sizeof(cl_uint4);
			applog(LOG_INFO, "Creating temp_X buffer of %lu bytes (%lu MB) for split kernels",
			       (unsigned long)temp_X_size, (unsigned long)(temp_X_size / (1024 * 1024)));
			
			// Create temp_X_buffer (Part 1 output, Part 2 input)
			clState->temp_X_buffer = clCreateBuffer(clState->context, CL_MEM_READ_WRITE, 
			                                        temp_X_size, NULL, &status);
			if (status != CL_SUCCESS) {
				applog(LOG_ERR, "Error %d: clCreateBuffer (temp_X_buffer), size: %lu bytes", 
				       status, (unsigned long)temp_X_size);
				applog(LOG_ERR, "Try reducing thread concurrency or disabling split kernels");
				return NULL;
			}
			
			// Create temp_X2_buffer (Part 2 output, Part 3 input)
			clState->temp_X2_buffer = clCreateBuffer(clState->context, CL_MEM_READ_WRITE, 
			                                         temp_X_size, NULL, &status);
			if (status != CL_SUCCESS || !clState->temp_X2_buffer) {
				applog(LOG_ERR, "Error %d: clCreateBuffer (temp_X2_buffer) failed, size: %lu bytes", 
				       status, (unsigned long)temp_X_size);
				clReleaseMemObject(clState->temp_X_buffer);
				applog(LOG_ERR, "Try reducing thread concurrency or disabling split kernels");
				return NULL;
			}
			applog(LOG_INFO, "temp_X and temp_X2 buffers created successfully");
		} else {
			clState->temp_X_buffer = NULL;
			clState->temp_X2_buffer = NULL;
		}
	} else
#endif
	clState->outputBuffer = clCreateBuffer(clState->context, CL_MEM_WRITE_ONLY, BUFFERSIZE, NULL, &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: clCreateBuffer (outputBuffer)", status);
		return NULL;
	}

	return clState;
}
#endif /* HAVE_OPENCL */

