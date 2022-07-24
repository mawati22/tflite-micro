/*
 * micro_speech_preprocess.h
 *
 *  Created on: Jun 23, 2022
 *      Author: bhanup
 */

#ifndef TENSORFLOW_LITE_MICRO_MICRO_SPEECH_PREPROCESS_H_
#define TENSORFLOW_LITE_MICRO_MICRO_SPEECH_PREPROCESS_H_

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
//#include "NatureDSP_types.h"
/* Common helper macros. */
//#include "common.h"
/* Signal Processing Library API. */
//#include "common_fpu.h"
#include <xtensa/tie/xt_misc.h>

#define MAX_CMD_LINE_LENGTH 512
#define MAX_MEM_ALLOCS      100
#define XA_MAX_ARGS         30
#define XA_SCREEN_WIDTH     80

#define FRAME_LENGTH 512
#define STRIDE_LENGTH 128
#define MAX_INT8 127
#define MIN_INT8 -128

extern char pb_input_file_path[];
extern char pb_output_file_path[];
extern float twd_512_f[];

#define NASSERT(x) {(void)__builtin_expect((x)!=0,1);}
#define NASSERT_ALIGN(addr,align) NASSERT(((uintptr_t)addr)%align==0)
#define NASSERT_ALIGN16(addr) NASSERT_ALIGN(addr,16)

#define NDSP_LIB_ENABLE 1
#define PROBE_DATA_TO_INVOKE 0
//Comparison enabled for python probe input to invoke
#if PROBE_DATA_TO_INVOKE
#define COMPARISON_ENABLED 1
#endif

//define the switch to get cycles information
#define __XT_PROFILE__ 0

#define WAV_HEADER_SIZE 44
#define Inv_sqrt_2 (0.70710678118)
#define Inv_sqrt_2_q15 23170 //(Inv_sqrt_2*(1<<15))
#define M_PI 3.14159265358979323846

#define complex_float  __complex__ float

#define MEM_ALLOC(mem_ptr, size, type) \
	mem_ptr = (type*)calloc(size, sizeof(type)); \
	if(mem_ptr == NULL) \
	{ \
		printf("memory allocation failed @%d\n", __LINE__); \
		exit(1); \
	}

#define FREE(mem_ptr) \
	free((void *)mem_ptr);

#define FILE_OPEN(file_ptr, file_name, mode) \
	file_ptr = fopen(file_name, mode); \
	if(file_ptr == NULL) \
	{ \
		printf("file open failed %s @%d\n", file_name, __LINE__); \
		exit(1); \
	}

#define FILE_CLOSE(file_ptr) \
		fclose(file_ptr);

#if COMPARISON_ENABLED
#define COMPARE_VAL(a, b, flag, sample_idx, frame) \
if(a != b) \
{ \
	  flag = 1; \
	  printf("Mismatch occurred @sample %d @frame %d, dut %d, ref %d\n", sample_idx, frame, a, b); \
} \
sample_idx++;

#define COMPARE_RESULT(flag) \
if (flag == 0) \
	printf("Test Result for python ref NS output comparison : Passed\n"); \
else \
	printf("Test Result for python ref NS output comparison : Failed\n");

#else
#define COMPARE_VAL(a, b, flag, sample_idx, frame)
#define COMPARE_RESULT(flag)
#endif

#define CLAMP_VAL(val, min, max) \
val = (val > max) ? max : val; \
val = (val < min) ? min : val;

#endif /*  */
