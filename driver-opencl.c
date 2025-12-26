/*
 * Copyright 2011-2012 Con Kolivas
 * Copyright 2011-2012 Luke Dashjr
 * Copyright 2010 Jeff Garzik
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include "config.h"

#ifdef HAVE_CURSES
#include <curses.h>
#endif

#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <sys/types.h>

#ifndef WIN32
#include <sys/resource.h>
#endif
#include <ccan/opt/opt.h>

#include "compat.h"
#include "miner.h"
#include "driver-opencl.h"
#include "findnonce.h"
#include "ocl.h"
#include "adl.h"
#include "util.h"

#ifdef USE_SCRYPT
#include "scrypt-jane.h"
#endif


/* TODO: cleanup externals ********************/

#ifdef HAVE_CURSES
extern WINDOW *mainwin, *statuswin, *logwin;
extern void enable_curses(void);
#endif

extern int mining_threads;
extern double total_secs;
extern int opt_g_threads;
extern bool opt_loginput;
extern char *opt_kernel_path;
extern int gpur_thr_id;
extern bool opt_noadl;
extern bool have_opencl;

extern void *miner_thread(void *userdata);
extern int dev_from_id(int thr_id);
extern void tailsprintf(char *f, const char *fmt, ...);
extern void decay_time(double *f, double fadd);

/**********************************************/

#ifdef HAVE_OPENCL
struct device_drv opencl_drv;
#endif

#ifdef HAVE_ADL
extern float gpu_temp(int gpu);
extern int gpu_fanspeed(int gpu);
extern int gpu_fanpercent(int gpu);
#endif

#ifdef HAVE_OPENCL
char *set_vector(char *arg)
{
	int i, val = 0, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set vector";
	val = atoi(nextptr);
	if (val != 1 && val != 2 && val != 4)
		return "Invalid value passed to set_vector";

	gpus[device++].vwidth = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);
		if (val != 1 && val != 2 && val != 4)
			return "Invalid value passed to set_vector";

		gpus[device++].vwidth = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].vwidth = gpus[0].vwidth;
	}

	return NULL;
}

char *set_worksize(char *arg)
{
	int i, val = 0, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set work size";
	val = atoi(nextptr);
	if (val < 1 || val > 9999)
		return "Invalid value passed to set_worksize";

	gpus[device++].work_size = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);
		if (val < 1 || val > 9999)
			return "Invalid value passed to set_worksize";

		gpus[device++].work_size = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].work_size = gpus[0].work_size;
	}

	return NULL;
}

#ifdef USE_SCRYPT
char *set_shaders(char *arg)
{
	int i, val = 0, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set lookup gap";
	val = atoi(nextptr);

	gpus[device++].shaders = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);

		gpus[device++].shaders = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].shaders = gpus[0].shaders;
	}

	return NULL;
}

char *set_lookup_gap(char *arg)
{
	int i, val = 0, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set lookup gap";
	val = atoi(nextptr);

	gpus[device++].opt_lg = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);

		gpus[device++].opt_lg = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].opt_lg = gpus[0].opt_lg;
	}

	return NULL;
}

char *set_thread_concurrency(char *arg)
{
	int i, val = 0, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set thread concurrency";
	val = atoi(nextptr);

	gpus[device++].opt_tc = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);

		gpus[device++].opt_tc = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].opt_tc = gpus[0].opt_tc;
	}

	return NULL;
}

char *set_buffer_size(char *arg)
{
	int i, val = 0, device = 0;
	char *nextptr;

	applog(LOG_DEBUG, "entering set_buffer_size");

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set buffer size";
	val = atoi(nextptr);

	gpus[device++].buffer_size = val;
	applog(LOG_DEBUG, "Buffer Size Set GPU %d: %d",device,val);

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);

		gpus[device++].buffer_size = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].buffer_size = gpus[0].buffer_size;
	}

	return NULL;
}

#endif

static enum cl_kernels select_kernel(char *arg)
{
	if (!strcmp(arg, "diablo"))
		return KL_DIABLO;
	if (!strcmp(arg, "diakgcn"))
		return KL_DIAKGCN;
	if (!strcmp(arg, "poclbm"))
		return KL_POCLBM;
	if (!strcmp(arg, "phatk"))
		return KL_PHATK;
#ifdef USE_SCRYPT
	if (!strcmp(arg, "scrypt"))
		return KL_SCRYPT;
	if (!strcmp(arg, "nscrypt"))
		return KL_N_SCRYPT;
	if (!strcmp(arg, "scrypt-chacha"))
		return KL_SCRYPT_CHACHA;
#endif
	return KL_NONE;
}

char *set_kernel(char *arg)
{
	enum cl_kernels kern;
	int i, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set kernel";
	kern = select_kernel(nextptr);
	if (kern == KL_NONE)
		return "Invalid parameter to set_kernel";
	gpus[device++].kernel = kern;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		kern = select_kernel(nextptr);
		if (kern == KL_NONE)
			return "Invalid parameter to set_kernel";

		gpus[device++].kernel = kern;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].kernel = gpus[0].kernel;
	}

	return NULL;
}
#endif

#ifdef HAVE_ADL
/* This function allows us to map an adl device to an opencl device for when
 * simple enumeration has failed to match them. */
char *set_gpu_map(char *arg)
{
	int val1 = 0, val2 = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set gpu map";
	if (sscanf(arg, "%d:%d", &val1, &val2) != 2)
		return "Invalid description for map pair";
	if (val1 < 0 || val1 > MAX_GPUDEVICES || val2 < 0 || val2 > MAX_GPUDEVICES)
		return "Invalid value passed to set_gpu_map";

	gpus[val1].virtual_adl = val2;
	gpus[val1].mapped = true;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		if (sscanf(nextptr, "%d:%d", &val1, &val2) != 2)
			return "Invalid description for map pair";
		if (val1 < 0 || val1 > MAX_GPUDEVICES || val2 < 0 || val2 > MAX_GPUDEVICES)
			return "Invalid value passed to set_gpu_map";
		gpus[val1].virtual_adl = val2;
		gpus[val1].mapped = true;
	}

	return NULL;
}
#endif

char *set_gpu_threads(char *arg)
{
	int i, val = 1, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set_gpu_threads";
	val = atoi(nextptr);
	if (val < 1 || val > 10)
		return "Invalid value passed to set_gpu_threads";

	gpus[device++].threads = val;
	applog(LOG_NOTICE,"Setting GPU %d threads to %d",device,val);

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);
		if (val < 1 || val > 10)
			return "Invalid value passed to set_gpu_threads";

		gpus[device++].threads = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].threads = gpus[0].threads;
	}

	return NULL;
}

#ifdef HAVE_ADL
char *set_gpu_engine(char *arg)
{
	int i, val1 = 0, val2 = 0, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set gpu engine";
	get_intrange(nextptr, &val1, &val2);
	if (val1 < 0 || val1 > 9999 || val2 < 0 || val2 > 9999)
		return "Invalid value passed to set_gpu_engine";

	gpus[device].min_engine = val1;
	gpus[device].gpu_engine = val2;
	device++;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		get_intrange(nextptr, &val1, &val2);
		if (val1 < 0 || val1 > 9999 || val2 < 0 || val2 > 9999)
			return "Invalid value passed to set_gpu_engine";
		gpus[device].min_engine = val1;
		gpus[device].gpu_engine = val2;
		device++;
	}

	if (device == 1) {
		for (i = 1; i < MAX_GPUDEVICES; i++) {
			gpus[i].min_engine = gpus[0].min_engine;
			gpus[i].gpu_engine = gpus[0].gpu_engine;
		}
	}

	return NULL;
}

char *set_gpu_fan(char *arg)
{
	int i, val1 = 0, val2 = 0, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set gpu fan";
	get_intrange(nextptr, &val1, &val2);
	if (val1 < 0 || val1 > 100 || val2 < 0 || val2 > 100)
		return "Invalid value passed to set_gpu_fan";

	gpus[device].min_fan = val1;
	gpus[device].gpu_fan = val2;
	device++;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		get_intrange(nextptr, &val1, &val2);
		if (val1 < 0 || val1 > 100 || val2 < 0 || val2 > 100)
			return "Invalid value passed to set_gpu_fan";

		gpus[device].min_fan = val1;
		gpus[device].gpu_fan = val2;
		device++;
	}

	if (device == 1) {
		for (i = 1; i < MAX_GPUDEVICES; i++) {
			gpus[i].min_fan = gpus[0].min_fan;
			gpus[i].gpu_fan = gpus[0].gpu_fan;
		}
	}

	return NULL;
}

char *set_gpu_memclock(char *arg)
{
	int i, val = 0, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set gpu memclock";
	val = atoi(nextptr);
	if (val < 0 || val >= 9999)
		return "Invalid value passed to set_gpu_memclock";

	gpus[device++].gpu_memclock = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);
		if (val < 0 || val >= 9999)
			return "Invalid value passed to set_gpu_memclock";

		gpus[device++].gpu_memclock = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].gpu_memclock = gpus[0].gpu_memclock;
	}

	return NULL;
}

char *set_gpu_memdiff(char *arg)
{
	int i, val = 0, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set gpu memdiff";
	val = atoi(nextptr);
	if (val < -9999 || val > 9999)
		return "Invalid value passed to set_gpu_memdiff";

	gpus[device++].gpu_memdiff = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);
		if (val < -9999 || val > 9999)
			return "Invalid value passed to set_gpu_memdiff";

		gpus[device++].gpu_memdiff = val;
	}
		if (device == 1) {
			for (i = device; i < MAX_GPUDEVICES; i++)
				gpus[i].gpu_memdiff = gpus[0].gpu_memdiff;
		}

			return NULL;
}

char *set_gpu_powertune(char *arg)
{
	int i, val = 0, device = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set gpu powertune";
	val = atoi(nextptr);
	if (val < -99 || val > 99)
		return "Invalid value passed to set_gpu_powertune";

	gpus[device++].gpu_powertune = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);
		if (val < -99 || val > 99)
			return "Invalid value passed to set_gpu_powertune";

		gpus[device++].gpu_powertune = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].gpu_powertune = gpus[0].gpu_powertune;
	}

	return NULL;
}

char *set_gpu_vddc(char *arg)
{
	int i, device = 0;
	float val = 0;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set gpu vddc";
	val = atof(nextptr);
	if (val < 0 || val >= 9999)
		return "Invalid value passed to set_gpu_vddc";

	gpus[device++].gpu_vddc = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atof(nextptr);
		if (val < 0 || val >= 9999)
			return "Invalid value passed to set_gpu_vddc";

		gpus[device++].gpu_vddc = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++)
			gpus[i].gpu_vddc = gpus[0].gpu_vddc;
	}

	return NULL;
}

char *set_temp_overheat(char *arg)
{
	int i, val = 0, device = 0, *to;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set temp overheat";
	val = atoi(nextptr);
	if (val < 0 || val > 200)
		return "Invalid value passed to set temp overheat";

	to = &gpus[device++].adl.overtemp;
	*to = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);
		if (val < 0 || val > 200)
			return "Invalid value passed to set temp overheat";

		to = &gpus[device++].adl.overtemp;
		*to = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++) {
			to = &gpus[i].adl.overtemp;
			*to = val;
		}
	}

	return NULL;
}

char *set_temp_target(char *arg)
{
	int i, val = 0, device = 0, *tt;
	char *nextptr;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set temp target";
	val = atoi(nextptr);
	if (val < 0 || val > 200)
		return "Invalid value passed to set temp target";

	tt = &gpus[device++].adl.targettemp;
	*tt = val;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		val = atoi(nextptr);
		if (val < 0 || val > 200)
			return "Invalid value passed to set temp target";

		tt = &gpus[device++].adl.targettemp;
		*tt = val;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++) {
			tt = &gpus[i].adl.targettemp;
			*tt = val;
		}
	}

	return NULL;
}
#endif
#ifdef HAVE_OPENCL
char *set_intensity(char *arg)
{
	int i, device = 0, *tt;
	char *nextptr, val = 0;

	nextptr = strtok(arg, ",");
	if (nextptr == NULL)
		return "Invalid parameters for set intensity";
	if (!strncasecmp(nextptr, "d", 1))
		gpus[device].dynamic = true;
	else {
		gpus[device].dynamic = false;
		val = atoi(nextptr);
		if (val < MIN_INTENSITY || val > MAX_INTENSITY)
			return "Invalid value passed to set intensity";
		tt = &gpus[device].intensity;
		*tt = val;
		gpus[device].xintensity = 0; // Disable shader based intensity
		gpus[device].rawintensity = 0; // Disable raw intensity
	}

	device++;

	while ((nextptr = strtok(NULL, ",")) != NULL) {
		if (!strncasecmp(nextptr, "d", 1))
			gpus[device].dynamic = true;
		else {
			gpus[device].dynamic = false;
			val = atoi(nextptr);
			if (val < MIN_INTENSITY || val > MAX_INTENSITY)
				return "Invalid value passed to set intensity";

			tt = &gpus[device].intensity;
			*tt = val;
			gpus[device].xintensity = 0; // Disable shader based intensity
			gpus[device].rawintensity = 0; // Disable raw intensity
		}
		device++;
	}
	if (device == 1) {
		for (i = device; i < MAX_GPUDEVICES; i++) {
			gpus[i].dynamic = gpus[0].dynamic;
			gpus[i].intensity = gpus[0].intensity;
			gpus[i].xintensity = 0; // Disable shader based intensity
			gpus[i].rawintensity = 0; // Disable raw intensity
		}
	}

	return NULL;
}

char *set_xintensity(char *arg)
{
  int i, device = 0, val = 0;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for shader based intensity";
  val = atoi(nextptr);
  if (val == 0) return "disabled";
  if (val < MIN_XINTENSITY || val > MAX_XINTENSITY)
    return "Invalid value passed to set shader intensity";

  gpus[device].dynamic = false; // Disable dynamic intensity
  gpus[device].intensity = 0; // Disable regular intensity
  gpus[device].rawintensity = 0; // Disable raw intensity
  gpus[device].xintensity = val;
  device++;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val < MIN_XINTENSITY || val > MAX_XINTENSITY)
      return "Invalid value passed to set shader based intensity";
    gpus[device].dynamic = false; // Disable dynamic intensity
    gpus[device].intensity = 0; // Disable regular intensity
    gpus[device].rawintensity = 0; // Disable raw intensity
    gpus[device].xintensity = val;
    device++;
  }
  if (device == 1)
    for (i = device; i < MAX_GPUDEVICES; i++) {
      gpus[i].dynamic = gpus[0].dynamic; // Disable dynamic intensity
      gpus[i].intensity = gpus[0].intensity; // Disable regular intensity
      gpus[i].rawintensity = gpus[0].rawintensity;
      gpus[i].xintensity = gpus[0].xintensity;
    }

  return NULL;
}

char *set_rawintensity(char *arg)
{
  int i, device = 0, val = 0;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for raw intensity";
  val = atoi(nextptr);
  if (val == 0) return "disabled";
  if (val < MIN_RAWINTENSITY || val > MAX_RAWINTENSITY)
    return "Invalid value passed to set raw intensity";

  gpus[device].dynamic = false; // Disable dynamic intensity
  gpus[device].intensity = 0; // Disable regular intensity
  gpus[device].xintensity = 0; // Disable xintensity
  gpus[device].rawintensity = val;
  device++;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val < MIN_RAWINTENSITY || val > MAX_RAWINTENSITY)
      return "Invalid value passed to set raw intensity";
    gpus[device].dynamic = false; // Disable dynamic intensity
    gpus[device].intensity = 0; // Disable regular intensity
    gpus[device].xintensity = 0; // Disable xintensity
    gpus[device].rawintensity = val;
    device++;
  }
  if (device == 1)
    for (i = device; i < MAX_GPUDEVICES; i++) {
      gpus[i].dynamic = gpus[0].dynamic;
      gpus[i].intensity = gpus[0].intensity;
      gpus[i].rawintensity = gpus[0].rawintensity;
      gpus[i].xintensity = gpus[0].xintensity;
    }

  return NULL;

}

void print_ndevs(int *ndevs)
{
	opt_log_output = true;
	opencl_drv.drv_detect();
	clear_adl(*ndevs);
	applog(LOG_INFO, "%i GPU devices max detected", *ndevs);
}
#endif

struct cgpu_info gpus[MAX_GPUDEVICES]; /* Maximum number apparently possible */
struct cgpu_info *cpus;

#ifdef HAVE_OPENCL

/* In dynamic mode, only the first thread of each device will be in use.
 * This potentially could start a thread that was stopped with the start-stop
 * options if one were to disable dynamic from the menu on a paused GPU */
void pause_dynamic_threads(int gpu)
{
	struct cgpu_info *cgpu = &gpus[gpu];
	int i;

	for (i = 1; i < cgpu->threads; i++) {
		struct thr_info *thr;

		thr = get_thread(i);
		if (!thr->pause && cgpu->dynamic) {
			applog(LOG_WARNING, "Disabling extra threads due to dynamic mode.");
			applog(LOG_WARNING, "Tune dynamic intensity with --gpu-dyninterval");
		}

		thr->pause = cgpu->dynamic;
		if (!cgpu->dynamic && cgpu->deven != DEV_DISABLED)
			cgsem_post(&thr->sem);
	}
}

#endif /* HAVE_OPENCL */

#if defined(HAVE_OPENCL) && defined(HAVE_CURSES)
void manage_gpu(void)
{
	struct thr_info *thr;
	int selected, gpu, i;
	char checkin[40];
	char input;

	if (!opt_g_threads) {
		applog(LOG_ERR, "opt_g_threads not set in manage_gpu()");
		return;
	}

	opt_loginput = true;
	immedok(logwin, true);
	clear_logwin();
retry:

	for (gpu = 0; gpu < nDevs; gpu++) {
		struct cgpu_info *cgpu = &gpus[gpu];
		double displayed_rolling, displayed_total;
		bool mhash_base = true;

		displayed_rolling = cgpu->rolling;
		displayed_total = cgpu->total_mhashes / total_secs;
		if (displayed_rolling < 1) {
			displayed_rolling *= 1000;
			displayed_total *= 1000;
			mhash_base = false;
		}

		wlog("GPU %d: %.1f / %.1f %sh/s | A:%d  R:%d  HW:%d  U:%.2f/m  I:%d xI:%d  rI:%d\n",
			gpu, displayed_rolling, displayed_total, mhash_base ? "M" : "K",
			cgpu->accepted, cgpu->rejected, cgpu->hw_errors,
			cgpu->utility, cgpu->intensity, cgpu->xintensity, cgpu->rawintensity);
#ifdef HAVE_ADL
		if (gpus[gpu].has_adl) {
			int engineclock = 0, memclock = 0, activity = 0, fanspeed = 0, fanpercent = 0, powertune = 0;
			float temp = 0, vddc = 0;

			if (gpu_stats(gpu, &temp, &engineclock, &memclock, &vddc, &activity, &fanspeed, &fanpercent, &powertune)) {
				char logline[255];

				strcpy(logline, ""); // In case it has no data
				if (temp != -1)
					sprintf(logline, "%.1f C  ", temp);
				if (fanspeed != -1 || fanpercent != -1) {
					tailsprintf(logline, "F: ");
					if (fanpercent != -1)
						tailsprintf(logline, "%d%% ", fanpercent);
					if (fanspeed != -1)
						tailsprintf(logline, "(%d RPM) ", fanspeed);
					tailsprintf(logline, " ");
				}
				if (engineclock != -1)
					tailsprintf(logline, "E: %d MHz  ", engineclock);
				if (memclock != -1)
					tailsprintf(logline, "M: %d Mhz  ", memclock);
				if (vddc != -1)
					tailsprintf(logline, "V: %.3fV  ", vddc);
				if (activity != -1)
					tailsprintf(logline, "A: %d%%  ", activity);
				if (powertune != -1)
					tailsprintf(logline, "P: %d%%", powertune);
				tailsprintf(logline, "\n");
				_wlog(logline);
			}
		}
#endif
		wlog("Last initialised: %s\n", cgpu->init);
		for (i = 0; i < mining_threads; i++) {
			thr = get_thread(i);
			if (thr->cgpu != cgpu)
				continue;
			get_datestamp(checkin, &thr->last);
			displayed_rolling = thr->rolling;
			if (!mhash_base)
				displayed_rolling *= 1000;
			wlog("Thread %d: %.1f %sh/s %s ", i, displayed_rolling, mhash_base ? "M" : "K" , cgpu->deven != DEV_DISABLED ? "Enabled" : "Disabled");
			switch (cgpu->status) {
				default:
				case LIFE_WELL:
					wlog("ALIVE");
					break;
				case LIFE_SICK:
					wlog("SICK reported in %s", checkin);
					break;
				case LIFE_DEAD:
					wlog("DEAD reported in %s", checkin);
					break;
				case LIFE_INIT:
				case LIFE_NOSTART:
					wlog("Never started");
					break;
			}
			if (thr->pause)
				wlog(" paused");
			wlog("\n");
		}
		wlog("\n");
	}

	wlogprint("[E]nable [D]isable [I]ntensity [x]Intensity R[a]w Intensity [R]estart GPU %s\n",adl_active ? "[C]hange settings" : "");

	wlogprint("Or press any other key to continue\n");
	logwin_update();
	input = getch();

	if (nDevs == 1)
		selected = 0;
	else
		selected = -1;
	if (!strncasecmp(&input, "e", 1)) {
		struct cgpu_info *cgpu;

		if (selected)
			selected = curses_int("Select GPU to enable");
		if (selected < 0 || selected >= nDevs) {
			wlogprint("Invalid selection\n");
			goto retry;
		}
		if (gpus[selected].deven != DEV_DISABLED) {
			wlogprint("Device already enabled\n");
			goto retry;
		}
		gpus[selected].deven = DEV_ENABLED;
		for (i = 0; i < mining_threads; ++i) {
			thr = get_thread(i);
			cgpu = thr->cgpu;
			if (cgpu->drv->drv_id != DRIVER_OPENCL)
				continue;
			if (dev_from_id(i) != selected)
				continue;
			if (cgpu->status != LIFE_WELL) {
				wlogprint("Must restart device before enabling it");
				goto retry;
			}
			applog(LOG_DEBUG, "Pushing sem post to thread %d", thr->id);

			cgsem_post(&thr->sem);
		}
		goto retry;
	} if (!strncasecmp(&input, "d", 1)) {
		if (selected)
			selected = curses_int("Select GPU to disable");
		if (selected < 0 || selected >= nDevs) {
			wlogprint("Invalid selection\n");
			goto retry;
		}
		if (gpus[selected].deven == DEV_DISABLED) {
			wlogprint("Device already disabled\n");
			goto retry;
		}
		gpus[selected].deven = DEV_DISABLED;
		goto retry;
	} else if (!strncasecmp(&input, "i", 1)) {
		int intensity;
		char *intvar;

		if (selected)
			selected = curses_int("Select GPU to change intensity on");
		if (selected < 0 || selected >= nDevs) {
			wlogprint("Invalid selection\n");
			goto retry;
		}
		intvar = curses_input("Set GPU scan intensity (d or " _MIN_INTENSITY_STR " -> " _MAX_INTENSITY_STR ")");
		if (!intvar) {
			wlogprint("Invalid input\n");
			goto retry;
		}
		if (!strncasecmp(intvar, "d", 1)) {
			wlogprint("Dynamic mode enabled on gpu %d\n", selected);
			gpus[selected].dynamic = true;
			pause_dynamic_threads(selected);
			free(intvar);
			goto retry;
		}
		intensity = atoi(intvar);
		free(intvar);
		if (intensity < MIN_INTENSITY || intensity > MAX_INTENSITY) {
			wlogprint("Invalid selection\n");
			goto retry;
		}
		gpus[selected].dynamic = false;
		gpus[selected].intensity = intensity;
		gpus[selected].xintensity = 0; // Disable xintensity when enabling intensity
		gpus[selected].rawintensity = 0; // Disable raw intensity when enabling intensity
		wlogprint("Intensity on gpu %d set to %d\n", selected, intensity);
		pause_dynamic_threads(selected);
		goto retry;
	} else if (!strncasecmp(&input, "x", 1)) {
		int xintensity;
		char *intvar;

		if (selected)
		selected = curses_int("Select GPU to change experimental intensity on");
		if (selected < 0 || selected >= nDevs) {
			wlogprint("Invalid selection\n");
			goto retry;
		}

		intvar = curses_input("Set experimental GPU scan intensity (" MIN_XINTENSITY_STR " -> " MAX_XINTENSITY_STR ")");
		if (!intvar) {
			wlogprint("Invalid input\n");
			goto retry;
		}
		xintensity = atoi(intvar);
		free(intvar);
		if (xintensity < MIN_XINTENSITY || xintensity > MAX_XINTENSITY) {
			wlogprint("Invalid selection\n");
			goto retry;
		}
		gpus[selected].dynamic = false;
		gpus[selected].intensity = 0; // Disable intensity when enabling xintensity
		gpus[selected].rawintensity = 0; // Disable raw intensity when enabling xintensity
		gpus[selected].xintensity = xintensity;
		wlogprint("Experimental intensity on gpu %d set to %d\n", selected, xintensity);
		pause_dynamic_threads(selected);
		goto retry;
	} else if (!strncasecmp(&input, "a", 1)) {
		int rawintensity;
		char *intvar;

		if (selected)
			selected = curses_int("Select GPU to change raw intensity on");
		if (selected < 0 || selected >= nDevs) {
			wlogprint("Invalid selection\n");
			goto retry;
		}

		intvar = curses_input("Set raw GPU scan intensity (" MIN_RAWINTENSITY_STR " -> " MAX_RAWINTENSITY_STR ")");
		if (!intvar) {
			wlogprint("Invalid input\n");
			goto retry;
		}
		rawintensity = atoi(intvar);
		free(intvar);
		if (rawintensity < MIN_RAWINTENSITY || rawintensity > MAX_RAWINTENSITY) {
			wlogprint("Invalid selection\n");
			goto retry;
		}
		gpus[selected].dynamic = false;
		gpus[selected].intensity = 0; // Disable intensity when enabling raw intensity
		gpus[selected].xintensity = 0; // Disable xintensity when enabling raw intensity
		gpus[selected].rawintensity = rawintensity;
		wlogprint("Raw intensity on gpu %d set to %d\n", selected, rawintensity);
		pause_dynamic_threads(selected);
		goto retry;
	} else if (!strncasecmp(&input, "r", 1)) {
		if (selected)
			selected = curses_int("Select GPU to attempt to restart");
		if (selected < 0 || selected >= nDevs) {
			wlogprint("Invalid selection\n");
			goto retry;
		}
		wlogprint("Attempting to restart threads of GPU %d\n", selected);
		reinit_device(&gpus[selected]);
		goto retry;
	} else if (adl_active && (!strncasecmp(&input, "c", 1))) {
		if (selected)
			selected = curses_int("Select GPU to change settings on");
		if (selected < 0 || selected >= nDevs) {
			wlogprint("Invalid selection\n");
			goto retry;
		}
		change_gpusettings(selected);
		goto retry;
	} else
		clear_logwin();

	immedok(logwin, false);
	opt_loginput = false;
}
#else
void manage_gpu(void)
{
}
#endif


#ifdef HAVE_OPENCL
static _clState *clStates[MAX_GPUDEVICES];

#define CL_SET_BLKARG(blkvar) status |= clSetKernelArg(*kernel, num++, sizeof(uint), (void *)&blk->blkvar)
#define CL_SET_ARG(var) status |= clSetKernelArg(*kernel, num++, sizeof(var), (void *)&var)
#define CL_SET_VARG(args, var) status |= clSetKernelArg(*kernel, num++, args * sizeof(uint), (void *)var)

static cl_int queue_poclbm_kernel(_clState *clState, dev_blk_ctx *blk, cl_uint threads)
{
	cl_kernel *kernel = &clState->kernel;
	unsigned int num = 0;
	cl_int status = 0;

	CL_SET_BLKARG(ctx_a);
	CL_SET_BLKARG(ctx_b);
	CL_SET_BLKARG(ctx_c);
	CL_SET_BLKARG(ctx_d);
	CL_SET_BLKARG(ctx_e);
	CL_SET_BLKARG(ctx_f);
	CL_SET_BLKARG(ctx_g);
	CL_SET_BLKARG(ctx_h);

	CL_SET_BLKARG(cty_b);
	CL_SET_BLKARG(cty_c);

	
	CL_SET_BLKARG(cty_f);
	CL_SET_BLKARG(cty_g);
	CL_SET_BLKARG(cty_h);

	if (!clState->goffset) {
		cl_uint vwidth = clState->vwidth;
		uint *nonces = alloca(sizeof(uint) * vwidth);
		unsigned int i;

		for (i = 0; i < vwidth; i++)
			nonces[i] = blk->nonce + (i * threads);
		CL_SET_VARG(vwidth, nonces);
	}

	CL_SET_BLKARG(fW0);
	CL_SET_BLKARG(fW1);
	CL_SET_BLKARG(fW2);
	CL_SET_BLKARG(fW3);
	CL_SET_BLKARG(fW15);
	CL_SET_BLKARG(fW01r);

	CL_SET_BLKARG(D1A);
	CL_SET_BLKARG(C1addK5);
	CL_SET_BLKARG(B1addK6);
	CL_SET_BLKARG(W16addK16);
	CL_SET_BLKARG(W17addK17);
	CL_SET_BLKARG(PreVal4addT1);
	CL_SET_BLKARG(PreVal0);

	CL_SET_ARG(clState->outputBuffer);

	return status;
}

static cl_int queue_phatk_kernel(_clState *clState, dev_blk_ctx *blk,
				 __maybe_unused cl_uint threads)
{
	cl_kernel *kernel = &clState->kernel;
	cl_uint vwidth = clState->vwidth;
	unsigned int i, num = 0;
	cl_int status = 0;
	uint *nonces;

	CL_SET_BLKARG(ctx_a);
	CL_SET_BLKARG(ctx_b);
	CL_SET_BLKARG(ctx_c);
	CL_SET_BLKARG(ctx_d);
	CL_SET_BLKARG(ctx_e);
	CL_SET_BLKARG(ctx_f);
	CL_SET_BLKARG(ctx_g);
	CL_SET_BLKARG(ctx_h);

	CL_SET_BLKARG(cty_b);
	CL_SET_BLKARG(cty_c);
	CL_SET_BLKARG(cty_d);
	CL_SET_BLKARG(cty_f);
	CL_SET_BLKARG(cty_g);
	CL_SET_BLKARG(cty_h);

	nonces = alloca(sizeof(uint) * vwidth);
	for (i = 0; i < vwidth; i++)
		nonces[i] = blk->nonce + i;
	CL_SET_VARG(vwidth, nonces);

	CL_SET_BLKARG(W16);
	CL_SET_BLKARG(W17);
	CL_SET_BLKARG(PreVal4_2);
	CL_SET_BLKARG(PreVal0);
	CL_SET_BLKARG(PreW18);
	CL_SET_BLKARG(PreW19);
	CL_SET_BLKARG(PreW31);
	CL_SET_BLKARG(PreW32);

	CL_SET_ARG(clState->outputBuffer);

	return status;
}

static cl_int queue_diakgcn_kernel(_clState *clState, dev_blk_ctx *blk,
				   __maybe_unused cl_uint threads)
{
	cl_kernel *kernel = &clState->kernel;
	unsigned int num = 0;
	cl_int status = 0;

	if (!clState->goffset) {
		cl_uint vwidth = clState->vwidth;
		uint *nonces = alloca(sizeof(uint) * vwidth);
		unsigned int i;
		for (i = 0; i < vwidth; i++)
			nonces[i] = blk->nonce + i;
		CL_SET_VARG(vwidth, nonces);
	}

	CL_SET_BLKARG(PreVal0);
	CL_SET_BLKARG(PreVal4_2);
	CL_SET_BLKARG(cty_h);
	CL_SET_BLKARG(D1A);
	CL_SET_BLKARG(cty_b);
	CL_SET_BLKARG(cty_c);
	CL_SET_BLKARG(cty_f);
	CL_SET_BLKARG(cty_g);
	CL_SET_BLKARG(C1addK5);
	CL_SET_BLKARG(B1addK6);
	CL_SET_BLKARG(PreVal0addK7);
	CL_SET_BLKARG(W16addK16);
	CL_SET_BLKARG(W17addK17);
	CL_SET_BLKARG(PreW18);
	CL_SET_BLKARG(PreW19);
	CL_SET_BLKARG(W16);
	CL_SET_BLKARG(W17);
	CL_SET_BLKARG(PreW31);
	CL_SET_BLKARG(PreW32);

	CL_SET_BLKARG(ctx_a);
	CL_SET_BLKARG(ctx_b);
	CL_SET_BLKARG(ctx_c);
	CL_SET_BLKARG(ctx_d);
	CL_SET_BLKARG(ctx_e);
	CL_SET_BLKARG(ctx_f);
	CL_SET_BLKARG(ctx_g);
	CL_SET_BLKARG(ctx_h);

	CL_SET_BLKARG(zeroA);
	CL_SET_BLKARG(zeroB);

	CL_SET_BLKARG(oneA);
	CL_SET_BLKARG(twoA);
	CL_SET_BLKARG(threeA);
	CL_SET_BLKARG(fourA);
	CL_SET_BLKARG(fiveA);
	CL_SET_BLKARG(sixA);
	CL_SET_BLKARG(sevenA);

	CL_SET_ARG(clState->outputBuffer);

	return status;
}

static cl_int queue_diablo_kernel(_clState *clState, dev_blk_ctx *blk, cl_uint threads)
{
	cl_kernel *kernel = &clState->kernel;
	unsigned int num = 0;
	cl_int status = 0;

	if (!clState->goffset) {
		cl_uint vwidth = clState->vwidth;
		uint *nonces = alloca(sizeof(uint) * vwidth);
		unsigned int i;

		for (i = 0; i < vwidth; i++)
			nonces[i] = blk->nonce + (i * threads);
		CL_SET_VARG(vwidth, nonces);
	}


	CL_SET_BLKARG(PreVal0);
	CL_SET_BLKARG(PreVal0addK7);
	CL_SET_BLKARG(PreVal4addT1);
	CL_SET_BLKARG(PreW18);
	CL_SET_BLKARG(PreW19);
	CL_SET_BLKARG(W16);
	CL_SET_BLKARG(W17);
	CL_SET_BLKARG(W16addK16);
	CL_SET_BLKARG(W17addK17);
	CL_SET_BLKARG(PreW31);
	CL_SET_BLKARG(PreW32);

	CL_SET_BLKARG(D1A);
	CL_SET_BLKARG(cty_b);
	CL_SET_BLKARG(cty_c);
	CL_SET_BLKARG(cty_h);
	CL_SET_BLKARG(cty_f);
	CL_SET_BLKARG(cty_g);

	CL_SET_BLKARG(C1addK5);
	CL_SET_BLKARG(B1addK6);

	CL_SET_BLKARG(ctx_a);
	CL_SET_BLKARG(ctx_b);
	CL_SET_BLKARG(ctx_c);
	CL_SET_BLKARG(ctx_d);
	CL_SET_BLKARG(ctx_e);
	CL_SET_BLKARG(ctx_f);
	CL_SET_BLKARG(ctx_g);
	CL_SET_BLKARG(ctx_h);

	CL_SET_ARG(clState->outputBuffer);

	return status;
}

#ifdef USE_SCRYPT

static inline void
be32enc_vect(uint32_t *dst, const uint32_t *src, uint32_t len)
{
	uint32_t i;

	for (i = 0; i < len; i++)
		dst[i] = bswap_32(src[i]);
}

static cl_int queue_scrypt_kernel(_clState *clState, dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
	cl_kernel *kernel = &clState->kernel;
	unsigned int num = 0;
	cl_uint le_target;
	cl_int status = 0;
	uint32_t data[21];

	int minn = sc_minn;
	int maxn = sc_maxn;
	long starttime = sc_starttime;
	cl_uint nfactor=10;
	cl_uint N;

	unsigned int timestamp;
	if (opt_scrypt_chacha_84) {
		// For 84-byte headers, timestamp is 8 bytes starting at offset 68
		// Use the lower 32 bits for timestamp
		timestamp = bswap_32(*((unsigned int *)(blk->work->data + 17*4)));
	} else {
		timestamp = bswap_32(*((unsigned int *)(blk->work->data + 17*4)));
	}

	if (opt_scrypt_chacha || opt_n_scrypt)
	{
		// set the global variable for use in the hashmeter
//		printf("about to print sc_minn");
		if (blk->work->pool->sc_minn)
			{
			minn = *blk->work->pool->sc_minn;
			//applog(LOG_NOTICE,"in queue_scrypt_kernel, blk->work->pool->sc_minn: %d",*blk->work->pool->sc_minn);
			}
		if (blk->work->pool->sc_maxn)
			{
			maxn = *blk->work->pool->sc_maxn;
			//applog(LOG_NOTICE,"in queue_scrypt_kernel, blk->work->pool->sc_maxn: %d",*blk->work->pool->sc_maxn);
			}
		if (blk->work->pool->sc_starttime)
			{
			starttime = *blk->work->pool->sc_starttime;
			//applog(LOG_NOTICE,"in queue_scrypt_kernel, blk->work->pool->sc_maxn: %d",*blk->work->pool->sc_starttime);
			}
		//sc_currentn = GetNfactor(timestamp);
		blk->work->pool->sc_lastnfactor = GetNfactor(timestamp, minn, maxn, starttime);
		sc_currentn = blk->work->pool->sc_lastnfactor;
        nfactor = blk->work->pool->sc_lastnfactor;
    
		N = (1 << (nfactor + 1));
	}

	le_target = *(cl_uint *)(blk->work->target + 28);

	int buffer_size = opt_scrypt_chacha_84 ? 84 : 80;
	if (!opt_scrypt_chacha) {
		clState->cldata = blk->work->data;
	} else {
		// Initialize the data array to zero
		memset(data, 0, sizeof(data));
		applog(LOG_DEBUG, "Timestamp: %d, Nfactor: %d, Target: %08x", timestamp, nfactor, le_target);
		if (opt_scrypt_chacha_84) {
			sj_be32enc_vect(data, (const uint32_t *)blk->work->data, 21);
		} else {
			sj_be32enc_vect(data, (const uint32_t *)blk->work->data, 20);
		}
		clState->cldata = data;
	}

	status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, buffer_size, clState->cldata, 0, NULL,NULL);

	CL_SET_ARG(clState->CLbuffer0);
	CL_SET_ARG(clState->outputBuffer);
	CL_SET_ARG(clState->padbuffer8);
	CL_SET_ARG(le_target);

	// If using the N Scrypt kernel, pass in NFactor
	if (clState->chosen_kernel == KL_N_SCRYPT)
		CL_SET_ARG(nfactor);

	return status;
}
#endif

// This is where the number of threads for the GPU gets set - originally 2^I
//static void set_threads_hashes(unsigned int vectors,int64_t *hashes, size_t *globalThreads,
//			       unsigned int minthreads, __maybe_unused int *intensity)
static void set_threads_hashes(unsigned int vectors, unsigned int compute_shaders, int64_t *hashes, size_t *globalThreads,
					unsigned int minthreads, __maybe_unused int *intensity, __maybe_unused int *xintensity, __maybe_unused int *rawintensity, size_t opt_tc)
{
	unsigned int threads = 0;

	/* Use thread-concurrency if set, otherwise use intensity-based calculation */
	if (opt_tc > 0) {
		threads = opt_tc;
	} else {
		while (threads < minthreads) {
			if (*rawintensity > 0) {
				threads = *rawintensity;
			} else if (*xintensity > 0) {
				threads = compute_shaders * *xintensity;
			} else {
				threads = 1 << *intensity;
			}
			if (threads < minthreads) {
				if (likely(*intensity < MAX_INTENSITY))
					(*intensity)++;
				else
					threads = minthreads;
			}
		}
	}

	*globalThreads = threads;
	*hashes = threads * vectors;
}
#endif /* HAVE_OPENCL */


#ifdef HAVE_OPENCL
/* We have only one thread that ever re-initialises GPUs, thus if any GPU
 * init command fails due to a completely wedged GPU, the thread will never
 * return, unable to harm other GPUs. If it does return, it means we only had
 * a soft failure and then the reinit_gpu thread is ready to tackle another
 * GPU */
void *reinit_gpu(void *userdata)
{
	struct thr_info *mythr = userdata;
	struct cgpu_info *cgpu;
	struct thr_info *thr;
	struct timeval now;
	char name[256];
	int thr_id;
	int gpu;

	pthread_detach(pthread_self());

select_cgpu:
	cgpu = tq_pop(mythr->q, NULL);
	if (!cgpu)
		goto out;

	if (clDevicesNum() != nDevs) {
		applog(LOG_WARNING, "Hardware not reporting same number of active devices, will not attempt to restart GPU");
		goto out;
	}

	gpu = cgpu->device_id;

	for (thr_id = 0; thr_id < mining_threads; ++thr_id) {
		thr = get_thread(thr_id);
		cgpu = thr->cgpu;
		if (cgpu->drv->drv_id != DRIVER_OPENCL)
			continue;
		if (dev_from_id(thr_id) != gpu)
			continue;

		thr = get_thread(thr_id);
		if (!thr) {
			applog(LOG_WARNING, "No reference to thread %d exists", thr_id);
			continue;
		}

		thr->rolling = thr->cgpu->rolling = 0;
		/* Reports the last time we tried to revive a sick GPU */
		cgtime(&thr->sick);
		if (!pthread_cancel(thr->pth)) {
			applog(LOG_WARNING, "Thread %d still exists, killing it off", thr_id);
		} else
			applog(LOG_WARNING, "Thread %d no longer exists", thr_id);
	}

	for (thr_id = 0; thr_id < mining_threads; ++thr_id) {
		int virtual_gpu;

		thr = get_thread(thr_id);
		cgpu = thr->cgpu;
		if (cgpu->drv->drv_id != DRIVER_OPENCL)
			continue;
		if (dev_from_id(thr_id) != gpu)
			continue;

		virtual_gpu = cgpu->virtual_gpu;
		/* Lose this ram cause we may get stuck here! */
		//tq_freeze(thr->q);

		thr->q = tq_new();
		if (!thr->q)
			quit(1, "Failed to tq_new in reinit_gpu");

		/* Lose this ram cause we may dereference in the dying thread! */
		//free(clState);

		applog(LOG_INFO, "Reinit GPU thread %d", thr_id);
		clStates[thr_id] = initCl(virtual_gpu, name, sizeof(name));
		if (!clStates[thr_id]) {
			applog(LOG_ERR, "Failed to reinit GPU thread %d", thr_id);
			goto select_cgpu;
		}
		applog(LOG_INFO, "initCl() finished. Found %s", name);

		if (unlikely(thr_info_create(thr, NULL, miner_thread, thr))) {
			applog(LOG_ERR, "thread %d create failed", thr_id);
			return NULL;
		}
		applog(LOG_WARNING, "Thread %d restarted", thr_id);
	}

	cgtime(&now);
	get_datestamp(cgpu->init, &now);

	for (thr_id = 0; thr_id < mining_threads; ++thr_id) {
		thr = get_thread(thr_id);
		cgpu = thr->cgpu;
		if (cgpu->drv->drv_id != DRIVER_OPENCL)
			continue;
		if (dev_from_id(thr_id) != gpu)
			continue;

		cgsem_post(&thr->sem);
	}

	goto select_cgpu;
out:
	return NULL;
}
#else
void *reinit_gpu(__maybe_unused void *userdata)
{
	return NULL;
}
#endif


#ifdef HAVE_OPENCL
static void opencl_detect()
{
	int i;

	nDevs = clDevicesNum();
	if (nDevs < 0) {
		applog(LOG_ERR, "clDevicesNum returned error, no GPUs usable");
		nDevs = 0;
	}

	if (!nDevs)
		return;

	/* If opt_g_threads is not set, use default 1 thread on scrypt and
	 * 2 for regular mining */
	if (opt_g_threads == -1) {
		if (opt_scrypt)
			opt_g_threads = 1;
		else
			opt_g_threads = 2;
	}

	if (opt_scrypt)
		opencl_drv.max_diff = 65536;

	for (i = 0; i < nDevs; ++i) {
		struct cgpu_info *cgpu;

		cgpu = &gpus[i];
		cgpu->deven = DEV_ENABLED;
		cgpu->drv = &opencl_drv;
		cgpu->device_id = i;
#ifndef HAVE_ADL
		cgpu->threads = opt_g_threads;
#else
		if (cgpu->threads < 1)
			cgpu->threads = 1;
#endif
		cgpu->virtual_gpu = i;
		add_cgpu(cgpu);
	}

	if (!opt_noadl)
		init_adl(nDevs);
}

static void reinit_opencl_device(struct cgpu_info *gpu)
{
	tq_push(control_thr[gpur_thr_id].q, gpu);
}

#ifdef HAVE_ADL
static void get_opencl_statline_before(char *buf, struct cgpu_info *gpu)
{
	if (gpu->has_adl) {
		int gpuid = gpu->device_id;
		float gt = gpu_temp(gpuid);
		int gf = gpu_fanspeed(gpuid);
		int gp;

		if (gt != -1)
			tailsprintf(buf, "%5.1fC ", gt);
		else
			tailsprintf(buf, "       ", gt);
		if (gf != -1)
			tailsprintf(buf, "%4dRPM ", gf);
		else if ((gp = gpu_fanpercent(gpuid)) != -1)
			tailsprintf(buf, "%3d%%    ", gp);
		else
			tailsprintf(buf, "        ");
		tailsprintf(buf, "| ");
	} else
		gpu->drv->get_statline_before = &blank_get_statline_before;
}
#endif

static void get_opencl_statline(char *buf, struct cgpu_info *gpu)
{
	if (gpu->rawintensity > 0)
		tailsprintf(buf, " T:%d rI:%4d", gpu->threads, gpu->rawintensity);
	else if (gpu->xintensity > 0)
		tailsprintf(buf, " T:%d xI:%3d", gpu->threads, gpu->xintensity);
	else
		tailsprintf(buf, " T:%d I:%2d", gpu->threads, gpu->intensity);
//	tailsprintf(buf, " I:%2d", gpu->intensity);
}

struct opencl_thread_data {
	cl_int (*queue_kernel_parameters)(_clState *, dev_blk_ctx *, cl_uint);
	uint32_t *res;
};

static uint32_t *blank_res;

static bool opencl_thread_prepare(struct thr_info *thr)
{
	char name[256];
	struct timeval now;
	struct cgpu_info *cgpu = thr->cgpu;
	int gpu = cgpu->device_id;
	int virtual_gpu = cgpu->virtual_gpu;
	int i = thr->id;
	static bool failmessage = false;
	int buffersize = opt_scrypt ? SCRYPT_BUFFERSIZE : BUFFERSIZE;

	if (!blank_res)
		blank_res = calloc(buffersize, 1);
	if (!blank_res) {
		applog(LOG_ERR, "Failed to calloc in opencl_thread_init");
		return false;
	}

	strcpy(name, "");
	applog(LOG_INFO, "Init GPU thread %i GPU %i virtual GPU %i", i, gpu, virtual_gpu);
	clStates[i] = initCl(virtual_gpu, name, sizeof(name));
	if (!clStates[i]) {
#ifdef HAVE_CURSES
		if (use_curses)
			enable_curses();
#endif
		applog(LOG_ERR, "Failed to init GPU thread %d, disabling device %d", i, gpu);
		if (!failmessage) {
			applog(LOG_ERR, "Restarting the GPU from the menu will not fix this.");
			applog(LOG_ERR, "Try restarting cgminer.");
			failmessage = true;
#ifdef HAVE_CURSES
			char *buf;
			if (use_curses) {
				buf = curses_input("Press enter to continue");
				if (buf)
					free(buf);
			}
#endif
		}
		cgpu->deven = DEV_DISABLED;
		cgpu->status = LIFE_NOSTART;

		dev_error(cgpu, REASON_DEV_NOSTART);

		return false;
	}
	if (!cgpu->name)
		cgpu->name = strdup(name);
	if (!cgpu->kname)
	{
		switch (clStates[i]->chosen_kernel) {
			case KL_DIABLO:
				cgpu->kname = "diablo";
				break;
			case KL_DIAKGCN:
				cgpu->kname = "diakgcn";
				break;
			case KL_PHATK:
				cgpu->kname = "phatk";
				break;
#ifdef USE_SCRYPT
			case KL_SCRYPT:
				cgpu->kname = "scrypt";
				break;
			case KL_N_SCRYPT:
				cgpu->kname = "nscrypt";
				break;
			case KL_SCRYPT_CHACHA:
				cgpu->kname = "scrypt-chacha";
				break;
#endif
			case KL_POCLBM:
				cgpu->kname = "poclbm";
				break;
			default:
				break;
		}
	}
	applog(LOG_INFO, "initCl() finished. Found %s", name);
	cgtime(&now);
	get_datestamp(cgpu->init, &now);

	have_opencl = true;

	return true;
}

static bool opencl_thread_init(struct thr_info *thr)
{
	const int thr_id = thr->id;
	struct cgpu_info *gpu = thr->cgpu;
	struct opencl_thread_data *thrdata;
	_clState *clState = clStates[thr_id];
	cl_int status = 0;
	thrdata = calloc(1, sizeof(*thrdata));
	thr->cgpu_data = thrdata;
	int buffersize = opt_scrypt ? SCRYPT_BUFFERSIZE : BUFFERSIZE;

	if (!thrdata) {
		applog(LOG_ERR, "Failed to calloc in opencl_thread_init");
		return false;
	}

	switch (clState->chosen_kernel) {
		case KL_POCLBM:
			thrdata->queue_kernel_parameters = &queue_poclbm_kernel;
			break;
		case KL_PHATK:
			thrdata->queue_kernel_parameters = &queue_phatk_kernel;
			break;
		case KL_DIAKGCN:
			thrdata->queue_kernel_parameters = &queue_diakgcn_kernel;
			break;
#ifdef USE_SCRYPT
		case KL_SCRYPT:
		case KL_N_SCRYPT:
		case KL_SCRYPT_CHACHA:
			thrdata->queue_kernel_parameters = &queue_scrypt_kernel;
			break;
#endif
		default:
		case KL_DIABLO:
			thrdata->queue_kernel_parameters = &queue_diablo_kernel;
			break;
	}

	thrdata->res = calloc(buffersize, 1);

	if (!thrdata->res) {
		free(thrdata);
		applog(LOG_ERR, "Failed to calloc in opencl_thread_init");
		return false;
	}

	status |= clEnqueueWriteBuffer(clState->commandQueue, clState->outputBuffer, CL_TRUE, 0,
				       buffersize, blank_res, 0, NULL, NULL);
	if (unlikely(status != CL_SUCCESS)) {
		applog(LOG_ERR, "Error: clEnqueueWriteBuffer failed.");
		return false;
	}

	gpu->status = LIFE_WELL;

	gpu->device_last_well = time(NULL);

	return true;
}


static bool opencl_prepare_work(struct thr_info __maybe_unused *thr, struct work *work)
{
#ifdef USE_SCRYPT
	if (opt_scrypt)
		work->blk.work = work;
	else
#endif
		precalc_hash(&work->blk, (uint32_t *)(work->midstate), (uint32_t *)(work->data + 64));
	return true;
}

extern int opt_dynamic_interval;

static int64_t opencl_scanhash(struct thr_info *thr, struct work *work,
				int64_t __maybe_unused max_nonce)
{
	const int thr_id = thr->id;
	struct opencl_thread_data *thrdata = thr->cgpu_data;
	struct cgpu_info *gpu = thr->cgpu;
	_clState *clState = clStates[thr_id];
	const cl_kernel *kernel = &clState->kernel;
	const int dynamic_us = opt_dynamic_interval * 1000;

	cl_int status;
	size_t globalThreads[1];
	size_t localThreads[1] = { clState->wsize };
	int64_t hashes;
	int found = opt_scrypt ? SCRYPT_FOUND : FOUND;
	int buffersize = opt_scrypt ? SCRYPT_BUFFERSIZE : BUFFERSIZE;

	// OpenCL profiling variables
	cl_event kernel_event = NULL;
	cl_event read_event = NULL;
	cl_event write_event = NULL;
	// For split kernels: track intermediate events for proper cleanup
	cl_event split_event_part1 = NULL;
	cl_event split_event_part2 = NULL;
	cl_ulong kernel_start_time = 0, kernel_end_time = 0;
	cl_ulong read_start_time = 0, read_end_time = 0;
	cl_ulong write_start_time = 0, write_end_time = 0;
	cl_ulong kernel_execution_time = 0;
	cl_ulong total_execution_time = 0;
	static int profiling_counter = 0;
	static double avg_kernel_time = 0.0;
	static double avg_total_time = 0.0;
	
	// Fallback timing mechanism
	struct timeval start_time, end_time;
	gettimeofday(&start_time, NULL);

	/* Windows' timer resolution is only 15ms so oversample 5x */
	if (gpu->dynamic && (++gpu->intervals * dynamic_us) > 70000) {
		struct timeval tv_gpuend;
		double gpu_us;

		cgtime(&tv_gpuend);
		gpu_us = us_tdiff(&tv_gpuend, &gpu->tv_gpustart) / gpu->intervals;
		if (gpu_us > dynamic_us) {
			if (gpu->intensity > MIN_INTENSITY)
				--gpu->intensity;
		} else if (gpu_us < dynamic_us / 2) {
			if (gpu->intensity < MAX_INTENSITY)
				++gpu->intensity;
		}
		memcpy(&(gpu->tv_gpustart), &tv_gpuend, sizeof(struct timeval));
		gpu->intervals = 0;
	}

	set_threads_hashes(clState->vwidth, clState->compute_shaders, &hashes, globalThreads, localThreads[0], &gpu->intensity, &gpu->xintensity, &gpu->rawintensity, gpu->opt_tc);
	if (hashes > gpu->max_hashes)
		gpu->max_hashes = hashes;

	// Check if we should use split kernels
	bool use_split = false;
#ifdef USE_SCRYPT
	use_split = (clState->use_split_kernels && opt_scrypt_chacha_84);
#endif
	
	if (use_split) {
		// ===== SPLIT KERNEL EXECUTION =====
		cl_event event_part1 = NULL, event_part2 = NULL, event_part3 = NULL;
		// Store references for cleanup later
		split_event_part1 = NULL;
		split_event_part2 = NULL;
		unsigned int num = 0;
		cl_uint le_target = *(cl_uint *)(work->target + 28);
		
		// Prepare input data (same as queue_scrypt_kernel does)
		uint32_t data[21];
		int minn = sc_minn;
		int maxn = sc_maxn;
		long starttime = sc_starttime;
		unsigned int timestamp;
		
		if (opt_scrypt_chacha_84) {
			// For 84-byte headers, timestamp is 8 bytes starting at offset 68
			// Use the lower 32 bits for timestamp
			timestamp = bswap_32(*((unsigned int *)(work->blk.work->data + 17*4)));
		} else {
			timestamp = bswap_32(*((unsigned int *)(work->blk.work->data + 17*4)));
		}
		
		if (opt_scrypt_chacha || opt_n_scrypt) {
			// Set the nfactor from pool settings
			if (work->blk.work->pool->sc_minn) {
				minn = *work->blk.work->pool->sc_minn;
			}
			if (work->blk.work->pool->sc_maxn) {
				maxn = *work->blk.work->pool->sc_maxn;
			}
			if (work->blk.work->pool->sc_starttime) {
				starttime = *work->blk.work->pool->sc_starttime;
			}
			work->blk.work->pool->sc_lastnfactor = GetNfactor(timestamp, minn, maxn, starttime);
			sc_currentn = work->blk.work->pool->sc_lastnfactor;
		}
		
		int buffer_size = opt_scrypt_chacha_84 ? 84 : 80;
		if (!opt_scrypt_chacha) {
			clState->cldata = work->blk.work->data;
		} else {
			// Initialize the data array to zero
			memset(data, 0, sizeof(data));
			applog(LOG_DEBUG, "Split kernel: Timestamp: %d, Nfactor: %d, Target: %08x", timestamp, sc_currentn, le_target);
			if (opt_scrypt_chacha_84) {
				sj_be32enc_vect(data, (const uint32_t *)work->blk.work->data, 21);
			} else {
				sj_be32enc_vect(data, (const uint32_t *)work->blk.work->data, 20);
			}
			clState->cldata = data;
		}
		
		// Write input data to CLbuffer0 (CRITICAL: without this, kernels read garbage!)
		status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, true, 0, buffer_size, clState->cldata, 0, NULL, NULL);
		if (unlikely(status != CL_SUCCESS)) {
			applog(LOG_ERR, "Error %d: clEnqueueWriteBuffer failed for split kernels.", status);
			return -1;
		}
		
		// Set arguments for Part 1: (input, temp_X)
		num = 0;
		status = clSetKernelArg(clState->kernel_part1, num++, sizeof(cl_mem), &clState->CLbuffer0);
		status |= clSetKernelArg(clState->kernel_part1, num++, sizeof(cl_mem), &clState->temp_X_buffer);
		if (unlikely(status != CL_SUCCESS)) {
			applog(LOG_ERR, "Error %d: clSetKernelArg Part 1 failed.", status);
			return -1;
		}
		
		// Launch Part 1
		if (clState->goffset) {
			size_t global_work_offset[1] = { work->blk.nonce };
			status = clEnqueueNDRangeKernel(clState->commandQueue, clState->kernel_part1, 1, 
			                                global_work_offset, globalThreads, localThreads, 
			                                0, NULL, &event_part1);
		} else {
			status = clEnqueueNDRangeKernel(clState->commandQueue, clState->kernel_part1, 1, NULL,
			                                globalThreads, localThreads, 0, NULL, &event_part1);
		}
		if (unlikely(status != CL_SUCCESS)) {
			applog(LOG_ERR, "Error %d: Enqueueing kernel Part 1 failed.", status);
			if (event_part1) clReleaseEvent(event_part1);
			return -1;
		}
		
		// Set arguments for Part 2: (temp_X input, temp_X2 output, padcache)
		num = 0;
		status = clSetKernelArg(clState->kernel_part2, num++, sizeof(cl_mem), &clState->temp_X_buffer);
		status |= clSetKernelArg(clState->kernel_part2, num++, sizeof(cl_mem), &clState->temp_X2_buffer);
		status |= clSetKernelArg(clState->kernel_part2, num++, sizeof(cl_mem), &clState->padbuffer8);
		if (unlikely(status != CL_SUCCESS)) {
			applog(LOG_ERR, "Error %d: clSetKernelArg Part 2 failed.", status);
			clReleaseEvent(event_part1);
			return -1;
		}
		
		// Launch Part 2 (wait for Part 1)
		if (clState->goffset) {
			size_t global_work_offset[1] = { work->blk.nonce };
			status = clEnqueueNDRangeKernel(clState->commandQueue, clState->kernel_part2, 1, 
			                                global_work_offset, globalThreads, localThreads,
			                                1, &event_part1, &event_part2);
		} else {
			status = clEnqueueNDRangeKernel(clState->commandQueue, clState->kernel_part2, 1, NULL,
			                                globalThreads, localThreads, 1, &event_part1, &event_part2);
		}
		if (unlikely(status != CL_SUCCESS)) {
			applog(LOG_ERR, "Error %d: Enqueueing kernel Part 2 failed.", status);
			clReleaseEvent(event_part1);
			if (event_part2) clReleaseEvent(event_part2);
			return -1;
		}
		
		// Set arguments for Part 3: (input, temp_X2, output, target)
		num = 0;
		status = clSetKernelArg(clState->kernel_part3, num++, sizeof(cl_mem), &clState->CLbuffer0);
		status |= clSetKernelArg(clState->kernel_part3, num++, sizeof(cl_mem), &clState->temp_X2_buffer);
		status |= clSetKernelArg(clState->kernel_part3, num++, sizeof(cl_mem), &clState->outputBuffer);
		status |= clSetKernelArg(clState->kernel_part3, num++, sizeof(cl_uint), &le_target);
		if (unlikely(status != CL_SUCCESS)) {
			applog(LOG_ERR, "Error %d: clSetKernelArg Part 3 failed.", status);
			clReleaseEvent(event_part1);
			clReleaseEvent(event_part2);
			return -1;
		}
		
		// Launch Part 3 (wait for Part 2)
		if (clState->goffset) {
			size_t global_work_offset[1] = { work->blk.nonce };
			status = clEnqueueNDRangeKernel(clState->commandQueue, clState->kernel_part3, 1, 
			                                global_work_offset, globalThreads, localThreads,
			                                1, &event_part2, &event_part3);
		} else {
			status = clEnqueueNDRangeKernel(clState->commandQueue, clState->kernel_part3, 1, NULL,
			                                globalThreads, localThreads, 1, &event_part2, &event_part3);
		}
		if (unlikely(status != CL_SUCCESS)) {
			applog(LOG_ERR, "Error %d: Enqueueing kernel Part 3 failed.", status);
			clReleaseEvent(event_part1);
			clReleaseEvent(event_part2);
			if (event_part3) clReleaseEvent(event_part3);
			return -1;
		}
		
		// Use event_part3 as the main kernel event for profiling
		kernel_event = event_part3;
		
		// Store references to intermediate events for cleanup after clFinish()
		// (OpenCL command queue maintains its own references via wait lists)
		split_event_part1 = event_part1;
		split_event_part2 = event_part2;
		
		applog(LOG_DEBUG, "Split kernels executed (Part 1 -> 2 -> 3)");
	} else {
		// ===== MONOLITHIC KERNEL EXECUTION =====
		status = thrdata->queue_kernel_parameters(clState, &work->blk, globalThreads[0]);
		if (unlikely(status != CL_SUCCESS)) {
			applog(LOG_ERR, "Error: clSetKernelArg of all params failed.");
			return -1;
		}

		// Enqueue kernel with profiling enabled
		if (clState->goffset) {
			size_t global_work_offset[1];

			global_work_offset[0] = work->blk.nonce;
			if (opt_scrypt_chacha)
			applog(LOG_DEBUG, "Nonce: %u, Global work size: %lu, local work size: %lu", work->blk.nonce, (unsigned long)globalThreads[0], (unsigned long)localThreads[0]);
			status = clEnqueueNDRangeKernel(clState->commandQueue, *kernel, 1, global_work_offset,
							globalThreads, localThreads, 0, NULL, &kernel_event);
		} else
			status = clEnqueueNDRangeKernel(clState->commandQueue, *kernel, 1, NULL,
							globalThreads, localThreads, 0, NULL, &kernel_event);
		if (unlikely(status != CL_SUCCESS)) {
			applog(LOG_ERR, "Error %d: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)", status);
			if (kernel_event) clReleaseEvent(kernel_event);
			return -1;
		}
	}

	// Enqueue read buffer with profiling (wait for kernel to complete)
	// For split kernels, kernel_event is event_part3; for monolithic, it's the kernel event
	// This ensures we don't read results before the kernels finish executing
	const cl_event *wait_list = kernel_event ? &kernel_event : NULL;
	const cl_uint num_wait_events = kernel_event ? 1 : 0;
	status = clEnqueueReadBuffer(clState->commandQueue, clState->outputBuffer, CL_FALSE, 0,
				     buffersize, thrdata->res, 
				     num_wait_events, wait_list, 
				     &read_event);
	if (unlikely(status != CL_SUCCESS)) {
		applog(LOG_ERR, "Error: clEnqueueReadBuffer failed error %d. (clEnqueueReadBuffer)", status);
		if (kernel_event) clReleaseEvent(kernel_event);
		if (read_event) clReleaseEvent(read_event);
		return -1;
	}

	if (opt_scrypt_chacha)
	{
		uint32_t target = *(uint32_t *)(work->target + 28);
		applog(LOG_DEBUG, "Nonce: %u, Target: %08x", work->blk.nonce, target);
	}
	
	/* The amount of work scanned can fluctuate when intensity changes
	 * and since we do this one cycle behind, we increment the work more
	 * than enough to prevent repeating work */
	work->blk.nonce += gpu->max_hashes;
	
	/* Check for nonce range overflow to prevent wrapping to other GPU ranges */
	if (total_devices > 1) {
		uint32_t nonce_range = 0xFFFFFFFF / total_devices;
		uint32_t max_nonce_for_gpu = (gpu->device_id + 1) * nonce_range - 1;
		if (work->blk.nonce > max_nonce_for_gpu) {
			applog(LOG_DEBUG, "GPU %d nonce range exhausted, resetting to start", gpu->device_id);
			work->blk.nonce = gpu->device_id * nonce_range;
		}
	}

	/* This finish flushes the readbuffer set with CL_FALSE in clEnqueueReadBuffer */
	clFinish(clState->commandQueue);

	// Get end time for fallback timing
	gettimeofday(&end_time, NULL);
	
	// Get profiling information
	if (kernel_event) {
		status = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, 
						sizeof(cl_ulong), &kernel_start_time, NULL);
		if (status == CL_SUCCESS) {
			status = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, 
							sizeof(cl_ulong), &kernel_end_time, NULL);
			if (status == CL_SUCCESS) {
				kernel_execution_time = kernel_end_time - kernel_start_time;
			}
		}
	}

	if (read_event) {
		status = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, 
						sizeof(cl_ulong), &read_start_time, NULL);
		if (status == CL_SUCCESS) {
			status = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, 
							sizeof(cl_ulong), &read_end_time, NULL);
			if (status == CL_SUCCESS) {
				total_execution_time = read_end_time - kernel_start_time;
			}
		}
	}
	
	// Use fallback timing if OpenCL profiling failed
	if (kernel_execution_time == 0) {
		long fallback_time_us = (end_time.tv_sec - start_time.tv_sec) * 1000000 + 
					(end_time.tv_usec - start_time.tv_usec);
		kernel_execution_time = fallback_time_us * 1000; // Convert to nanoseconds
		total_execution_time = kernel_execution_time;
	}

	// Calculate averages and log profiling info every iteration
	profiling_counter++;
	if (profiling_counter == 1) {
		avg_kernel_time = kernel_execution_time;
		avg_total_time = total_execution_time;
	} else {
		avg_kernel_time = (avg_kernel_time * (profiling_counter - 1) + kernel_execution_time) / profiling_counter;
		avg_total_time = (avg_total_time * (profiling_counter - 1) + total_execution_time) / profiling_counter;
	}

	// Log profiling info every iteration
	double kernel_time_ms = kernel_execution_time / 1000000.0; // Convert ns to ms
	double total_time_ms = total_execution_time / 1000000.0;
	double avg_kernel_ms = avg_kernel_time / 1000000.0;
	double avg_total_ms = avg_total_time / 1000000.0;
	
	// Calculate memory bandwidth and occupancy estimates
	size_t memory_transferred = buffersize + (opt_scrypt_chacha_84 ? 84 : 80);
	double bandwidth_gbps = 0.0;
	if (total_time_ms > 0.001) { // Avoid division by very small numbers
		bandwidth_gbps = (memory_transferred / (total_time_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
	}
	
	// Estimate occupancy based on work group size and execution time
	double estimated_occupancy = (localThreads[0] * 100.0) / 64.0; // Assuming 64 work items per CU
	if (estimated_occupancy > 100.0) estimated_occupancy = 100.0;
	
	// Determine timing method used
	const char* timing_method = (kernel_start_time == 0) ? " [Fallback]" : " [OpenCL]";
	
	applog(LOG_INFO, "GPU %d Profiling [%d]: Kernel: %.2fms (avg: %.2fms), Total: %.2fms (avg: %.2fms), "
	       "Work Items: %lu, Memory: %.2fGB/s, Est. Occupancy: %.1f%%, Private Mem: ~2.4KB/workitem%s",
	       gpu->device_id, profiling_counter, kernel_time_ms, avg_kernel_ms, total_time_ms, avg_total_ms,
	       (unsigned long)globalThreads[0], bandwidth_gbps, estimated_occupancy, timing_method);
	
	// Check for potential memory spilling indicators
	if (kernel_time_ms > avg_kernel_ms * 1.5) {
		applog(LOG_WARNING, "GPU %d: Kernel execution time spike detected (%.2fms vs avg %.2fms) - possible memory spilling",
		       gpu->device_id, kernel_time_ms, avg_kernel_ms);
	}
	
	if (estimated_occupancy < 25.0) {
		applog(LOG_WARNING, "GPU %d: Low estimated occupancy (%.1f%%) - consider reducing private memory usage",
		       gpu->device_id, estimated_occupancy);
	}

	/* FOUND entry is used as a counter to say how many nonces exist */
	if (thrdata->res[found]) {
		/* Clear the buffer again */
		status = clEnqueueWriteBuffer(clState->commandQueue, clState->outputBuffer, CL_FALSE, 0,
					      buffersize, blank_res, 0, NULL, &write_event);
		if (unlikely(status != CL_SUCCESS)) {
			applog(LOG_ERR, "Error: clEnqueueWriteBuffer failed.");
			if (kernel_event) clReleaseEvent(kernel_event);
			if (read_event) clReleaseEvent(read_event);
			if (write_event) clReleaseEvent(write_event);
			return -1;
		}
		applog(LOG_DEBUG, "GPU %d found something?", gpu->device_id);
		postcalc_hash_async(thr, work, thrdata->res);
		memset(thrdata->res, 0, buffersize);
		/* This finish flushes the writebuffer set with CL_FALSE in clEnqueueWriteBuffer */
		clFinish(clState->commandQueue);
	}

	// Clean up events (after clFinish ensures all operations are complete)
	if (kernel_event) clReleaseEvent(kernel_event);
	if (read_event) clReleaseEvent(read_event);
	if (write_event) clReleaseEvent(write_event);
	
	// Clean up split kernel intermediate events (they're only needed for dependency chaining)
	if (use_split) {
		if (split_event_part1) clReleaseEvent(split_event_part1);
		if (split_event_part2) clReleaseEvent(split_event_part2);
	}

	return hashes;
}

static void opencl_thread_shutdown(struct thr_info *thr)
{
	const int thr_id = thr->id;
	_clState *clState = clStates[thr_id];

	// Release split kernels if they were created
#ifdef USE_SCRYPT
	if (clState->use_split_kernels) {
		if (clState->kernel_part1) clReleaseKernel(clState->kernel_part1);
		if (clState->kernel_part2) clReleaseKernel(clState->kernel_part2);
		if (clState->kernel_part3) clReleaseKernel(clState->kernel_part3);
		if (clState->temp_X_buffer) clReleaseMemObject(clState->temp_X_buffer);
		if (clState->temp_X2_buffer) clReleaseMemObject(clState->temp_X2_buffer);
		applog(LOG_DEBUG, "Released split kernel resources");
	}
#endif
	
	// Release monolithic kernel
	if (clState->kernel) clReleaseKernel(clState->kernel);
	clReleaseProgram(clState->program);
	clReleaseCommandQueue(clState->commandQueue);
	clReleaseContext(clState->context);
}

struct device_drv opencl_drv = {
	.drv_id = DRIVER_OPENCL,
	.dname = "opencl",
	.name = "GPU",
	.drv_detect = opencl_detect,
	.reinit_device = reinit_opencl_device,
#ifdef HAVE_ADL
	.get_statline_before = get_opencl_statline_before,
#endif
	.get_statline = get_opencl_statline,
	.thread_prepare = opencl_thread_prepare,
	.thread_init = opencl_thread_init,
	.prepare_work = opencl_prepare_work,
	.scanhash = opencl_scanhash,
	.thread_shutdown = opencl_thread_shutdown,
};
#endif
