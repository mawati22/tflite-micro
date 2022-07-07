/*
 * micro_speech_preprocess.h
 *
 *  Created on: Jun 23, 2022
 *      Author: bhanup
 */

#ifndef TENSORFLOW_LITE_MICRO_MICRO_SPEECH_PREPROCESS_H_
#define TENSORFLOW_LITE_MICRO_MICRO_SPEECH_PREPROCESS_H_

#define MAX_CMD_LINE_LENGTH 512
#define MAX_MEM_ALLOCS      100
#define XA_MAX_ARGS         30
#define XA_SCREEN_WIDTH     80

#define FRAME_LENGTH 512
#define STRIDE_LENGTH 128

extern char pb_input_file_path[];
extern char pb_output_file_path[];
extern float twd_512_f[];

//#include "NatureDSP_types.h"
/* Common helper macros. */
#include "common.h"
/* Signal Processing Library API. */
//#include "NatureDSP_Signal_fft.h"
#include "common_fpu.h"
#include <xtensa/tie/xt_misc.h>
#include <complex.h>
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

#define WAV_HEADER_SIZE 22
#define PROBE_DATA_SIZE (1247*257)

union complex_mag {
    complex_float * complex;
    float * mixed ;
};

#define MEM_ALLOC(mem_ptr, size, type) \
	mem_ptr = (type*)calloc(size, sizeof(type)); \
	if(mem_ptr == NULL) \
	{ \
		printf("memory allocation failed @%d\n", __LINE__); \
		exit(1); \
	}

#define FREE(mem_ptr) \
	free(mem_ptr);

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

#endif

#define complex_float  __complex__ float

void prefix_inpdir_name(char *inp_file);
void prefix_outdir_name(char *out_file);
void set_wbna(int *argc, char *argv[]);
int16_t FloatToFixed_Q1_15_Sat(float input);
void CalculateDiscreteFourierTransform(float* time_series, int time_series_size, float* fourier_output);
void CalculateInverseDiscreteFourierTransform(float* fourier_input, int fourier_input_size, float* time_series);
void complex_to_mag(float32_t  * restrict y, const complex_float  * restrict x, int N);
void vec_complex2mag (float32_t  * restrict y, const complex_float  * restrict x, int N);
int fft_realf_ie(complex_float* y,float32_t* x, const complex_float* twd, int twdstep, int N);
int ifft_realf_ie(float32_t* y, complex_float* x, const  complex_float* twd, int twdstep, int N);
void vec_cosinef(float32_t* restrict y, const float32_t* restrict x, int N);
void vec_sinef(float32_t* restrict y, const float32_t* restrict x, int N);
void vec_atan2f( float32_t* restrict z, const float32_t* restrict y, const float32_t* restrict x, int N);
void FloatToFixed_Q1_15_Sat_vec(float *input, int16_t *output, int N);

#endif /*  */
