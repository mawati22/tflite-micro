/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/micro/examples/micro_speech/dtln_noise_suppression_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/examples/micro_speech/dtln_noise_suppression.h"

#ifndef PARAMFILE
#define PARAMFILE "paramfilesimple.txt"
#endif

extern "C" int test_rfft_32x32_512(int *out, int *inp);
extern "C" int test_irfft_32x32_512(float *out, float *inp, int byteCount);
extern "C" int test_vec_atan2f(float32_t *output_ang, float32_t *i_value_f, float32_t *r_value_f, int bytesCount);
extern "C" int test_vec_complex2mag (float32_t  * restrict y, const complex_float  * restrict x, int N);
extern "C" int test_vec_cosinef(float32_t *real_f, const float32_t *output_ang, int bytesCount);
extern "C" int test_vec_sinef(float32_t *imag_f, const float32_t *output_ang, int bytesCount);
extern "C" int test_vec_int2float(float32_t *fft_out_f, const int32_t *fft_out, int scale2, int N);
extern "C" int FloatToFixed_Q1_15_Sat_vec(float *input, short *output, int N);
extern "C" int test_fft_real_512(short *preinput, float *fft_out_f, int N);
extern "C" int test_ifft_real_512(float *out, float *inp, int N);

char pb_input_file_path[MAX_CMD_LINE_LENGTH];
char pb_output_file_path[MAX_CMD_LINE_LENGTH];
char pb_input_file_name[MAX_CMD_LINE_LENGTH];
char pb_output_file_name[MAX_CMD_LINE_LENGTH];
int frame_num = 0;

union complex_mag {
    complex_float *complex;
    float *mixed ;
};

typedef struct wavfile_header_s
{
    char    ChunkID[4];     /*  4   */
    int32_t ChunkSize;      /*  4   */
    char    Format[4];      /*  4   */

    char    Subchunk1ID[4]; /*  4   */
    int32_t Subchunk1Size;  /*  4   */
    int16_t AudioFormat;    /*  2   */
    int16_t NumChannels;    /*  2   */
    int32_t SampleRate;     /*  4   */
    int32_t ByteRate;       /*  4   */
    int16_t BlockAlign;     /*  2   */
    int16_t BitsPerSample;  /*  2   */

    char    Subchunk2ID[4];
    int32_t Subchunk2Size;
} wavfile_header_t;

#define SUBCHUNK1SIZE   (16)
#define AUDIO_FORMAT    (1) /*For PCM*/
#define NUM_CHANNELS    (1)
#define SAMPLE_RATE     (16000)
#define BITS_PER_SAMPLE (16)
#define BYTE_RATE       (SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE / 8)
#define BLOCK_ALIGN     (NUM_CHANNELS * BITS_PER_SAMPLE / 8)

int write_PCM16_mono_header(FILE *file_p,
                                int32_t SampleRate,
                                int32_t size)
{
    int ret;

    fseek(file_p, 0, SEEK_SET);
    wavfile_header_t wav_header;
    int32_t subchunk2_size;
    int32_t chunk_size;
    
    size_t write_count;
    
    subchunk2_size  = size * NUM_CHANNELS * BITS_PER_SAMPLE / 8;
    chunk_size      = 4 + (8 + SUBCHUNK1SIZE) + (8 + subchunk2_size);
    
    wav_header.ChunkID[0] = 'R';
    wav_header.ChunkID[1] = 'I';
    wav_header.ChunkID[2] = 'F';
    wav_header.ChunkID[3] = 'F';
    
    wav_header.ChunkSize = chunk_size;
    
    wav_header.Format[0] = 'W';
    wav_header.Format[1] = 'A';
    wav_header.Format[2] = 'V';
    wav_header.Format[3] = 'E';
    
    wav_header.Subchunk1ID[0] = 'f';
    wav_header.Subchunk1ID[1] = 'm';
    wav_header.Subchunk1ID[2] = 't';
    wav_header.Subchunk1ID[3] = ' ';
    
    wav_header.Subchunk1Size = SUBCHUNK1SIZE;
    wav_header.AudioFormat = AUDIO_FORMAT;
    wav_header.NumChannels = NUM_CHANNELS;
    wav_header.SampleRate = SampleRate;
    wav_header.ByteRate = BYTE_RATE;
    wav_header.BlockAlign = BLOCK_ALIGN;
    wav_header.BitsPerSample = BITS_PER_SAMPLE;
    
    wav_header.Subchunk2ID[0] = 'd';
    wav_header.Subchunk2ID[1] = 'a';
    wav_header.Subchunk2ID[2] = 't';
    wav_header.Subchunk2ID[3] = 'a';
    wav_header.Subchunk2Size = subchunk2_size;

    write_count = fwrite(   &wav_header, 
                            sizeof(wavfile_header_t), 1,
                            file_p);

    ret = (1 != write_count)? -1 : 0;
    
    return ret;
}

static void prefix_dirname(char *fname, char *dirname)
{
    if(strlen(dirname) > 0)
    {
        char tempname[MAX_CMD_LINE_LENGTH];
        strncpy(tempname, fname, MAX_CMD_LINE_LENGTH);
        // Remove additional / slash from directory name
        if(dirname[strlen(dirname)-1] == '/')
        {
            dirname[strlen(dirname)-1] = '\0';
        }
        snprintf(fname,MAX_CMD_LINE_LENGTH, "%s/%s", dirname, tempname);
    }
}

void prefix_inpdir_name(char *inp_file)
{
    prefix_dirname(inp_file, pb_input_file_path);
}

void prefix_outdir_name(char *out_file)
{
    prefix_dirname(out_file, pb_output_file_path);
}

// Set cache attribute to Write Back No Allocate when the last argument is -wbna
void set_wbna(int *argc, char *argv[])
{
    if ( *argc > 1 && !strcmp(argv[*argc-1], "-wbna") )
    {
#ifdef __XCC__
        extern char _memmap_cacheattr_wbna_trapnull;

        xthal_set_cacheattr((unsigned)&_memmap_cacheattr_wbna_trapnull);
#endif
        (*argc)--;
    }
}

TF_LITE_MICRO_TESTS_BEGIN

  FILE *param_file_id = NULL;
  char curr_cmd[MAX_CMD_LINE_LENGTH] = {0};
  char fargv[XA_MAX_ARGS][MAX_CMD_LINE_LENGTH] = {{0}};
  int fargc = 0, curpos = 0, processcmd = 0;
#if COMPARISON_ENABLED
  int is_mismatch_occured = 0;
#endif

  memset(pb_input_file_path,  0, MAX_CMD_LINE_LENGTH);
  memset(pb_output_file_path, 0, MAX_CMD_LINE_LENGTH);

  // NOTE: set_wbna() should be called before any other dynamic
  // adjustment of the region attributes for cache.
  set_wbna(&argc, argv);

#if PROBE_DATA_TO_INVOKE
#if COMPARISON_ENABLED
  if(argc<5)
  {
      printf("Usage: %s <fullpath-input.wav> <fullpath-output.wav> <fullpath-invoke1-ref-input.txt> <fullpath-invoke1-ref-output.txt>\n", argv[0]);
#else
  if(argc<4)
  {
      printf("Usage: %s <fullpath-input.wav> <fullpath-output.wav> <fullpath-invoke1-ref-input.txt> \n", argv[0]);
#endif
#else
  if(argc<3)
  {
	  printf("Usage: %s <fullpath-input.wav> <fullpath-output.wav> \n", argv[0]);
#endif
	  printf("Going ahead with default PARAMFILE.\n");
      if ((param_file_id = fopen(PARAMFILE, "r")) == NULL )
      {
        printf("Parameter file \"%s\" not found.\n", PARAMFILE);
        exit(0);
       }
  }

TF_LITE_MICRO_TEST(TestInvoke)
{
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(dtln_NS_model);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  tflite::MicroMutableOpResolver<4> micro_op_resolver;
  micro_op_resolver.AddUnidirectionalSequenceLSTM();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddReshape();

  // Create an area of memory to use for input, output, and intermediate arrays.
#if defined(XTENSA)
  constexpr int tensor_arena_size = 100 * 1024;
#else
  constexpr int tensor_arena_size = 10 * 1024;
#endif
  uint8_t tensor_arena1[tensor_arena_size];

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena1,
                                       tensor_arena_size,
                                       &micro_error_reporter);

  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  // Make sure the input has the properties we expect.
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, input->type);

  int bytesCount = (int)(input->bytes);

  float *output_mag=NULL, *r_value_f = NULL, *i_value_f = NULL, *output_ang = NULL;
  float *model_output_f = NULL, *predicted_mag_f = NULL;
  float *real_f = NULL, *imag_f = NULL, *ifft_input = NULL;
  float *ifft_output = NULL;
  float *final_output = NULL;
  int16_t *output_int16 = NULL;
  float *overlap_add_buf = NULL, *preinput_f=NULL;
  short *preinput=NULL;
  float *fft_out_f = NULL;
#if PROBE_DATA_TO_INVOKE
  short *inject_in=NULL;
#endif
#if COMPARISON_ENABLED
  short *ref_out_NS = NULL;
#endif
  //Modify the size FRAME_LENGTH+STRIDE_LENGTH
  MEM_ALLOC(overlap_add_buf, 6*STRIDE_LENGTH, float);
  MEM_ALLOC(preinput, FRAME_LENGTH, int16_t);
  MEM_ALLOC(preinput_f, FRAME_LENGTH, float);
  MEM_ALLOC(output_mag, bytesCount, float);
  MEM_ALLOC(r_value_f, bytesCount, float);
  MEM_ALLOC(i_value_f, bytesCount, float);
  MEM_ALLOC(output_ang, bytesCount, float);
  MEM_ALLOC(model_output_f, bytesCount, float);
  MEM_ALLOC(predicted_mag_f, bytesCount, float);
  MEM_ALLOC(real_f, bytesCount, float);
  MEM_ALLOC(imag_f, bytesCount, float);
  MEM_ALLOC(ifft_input, (FRAME_LENGTH+2), float);
  MEM_ALLOC(ifft_output, FRAME_LENGTH, float);
  MEM_ALLOC(final_output, STRIDE_LENGTH, float);
  MEM_ALLOC(output_int16, STRIDE_LENGTH, int16_t);
  MEM_ALLOC(fft_out_f, FRAME_LENGTH+2, float);

  if (param_file_id == NULL)
  {
    /* Process only one commandline input */
    FILE *fp, *fout;
    FILE_OPEN(fp, argv[1], "rb");
    strncpy(pb_output_file_name, argv[2],MAX_CMD_LINE_LENGTH);
    FILE_OPEN(fout, argv[2], "wb");
    printf("Processing input file: '%s'\n", argv[1]);
    fseek(fp, 0, SEEK_END);
    long lSize = ftell (fp) - WAV_HEADER_SIZE; //find total samples
    rewind (fp);
    //int num_samples = (int)floor(lSize/sizeof(int16_t));
	int total_frames = (int)floor(lSize/(STRIDE_LENGTH*sizeof(int16_t)));

	write_PCM16_mono_header(fout, SAMPLE_RATE, total_frames*STRIDE_LENGTH);

#if PROBE_DATA_TO_INVOKE
    printf("Injecting Invoke1 input from Python Ref...\n");
	FILE *finject;
	FILE_OPEN(finject, argv[3], "r");
	MEM_ALLOC(inject_in, total_frames*bytesCount, short);
    static int sample_index = 0;
    int neg = 0;
    char ch;
    sample_index = 0;
    do {
       ch = fgetc(finject);
       if( feof(finject) ) {
          break ;
       }
       if(ch == ',')
       {
           if(neg == 1)
               inject_in[sample_index] *= -1;
           sample_index++;
           neg = 0;
       }
       else if(ch == '-')
           neg = 1;
       else
           inject_in[sample_index] = inject_in[sample_index]*10 + (short)(ch - '0');
    } while(1);
    FILE_CLOSE(finject);
    printf("Injecting Done.\n");
    sample_index = 0;
#endif //#if PROBE_DATA_TO_INVOKE
#if COMPARISON_ENABLED
    FILE *frefout;
    FILE_OPEN(frefout, argv[4], "r");
    MEM_ALLOC(ref_out_NS, total_frames*bytesCount, short);
    static int sample_out_index_cnt = 0;
    neg = 0;
    sample_out_index_cnt = 0;
    do {
        ch = fgetc(frefout);
        if( feof(frefout) ) {
           break ;
        }
        if(ch == ',')
        {
            if(neg == 1)
                ref_out_NS[sample_out_index_cnt] *= -1;
            sample_out_index_cnt++;
            neg = 0;
        }
        else if(ch == '-')
            neg = 1;
        else
            ref_out_NS[sample_out_index_cnt] = ref_out_NS[sample_out_index_cnt]*10 + (short)(ch - '0');
    } while(1);
    FILE_CLOSE(frefout);
    sample_out_index_cnt = 0;
#endif
    /*  Main While loop */
	for(int frm=0; frm<total_frames; frm++) {
    fseek(fp, WAV_HEADER_SIZE+(frm*STRIDE_LENGTH*sizeof(int16_t)), SEEK_SET);
    int read_samples = fread(preinput, sizeof(int16_t), (size_t)FRAME_LENGTH, fp);
	if(read_samples < FRAME_LENGTH)
	{
#if PROBE_DATA_TO_INVOKE
        break;
#endif
		for(int i=read_samples; i<FRAME_LENGTH; i++)
			preinput[i] = 0;
	}
    // Pre-processing code (STFT)
	test_fft_real_512(preinput, fft_out_f, FRAME_LENGTH);

	complex_mag cmolx_fourier;
    cmolx_fourier.mixed = fft_out_f;
    test_vec_complex2mag(output_mag, cmolx_fourier.complex, bytesCount);

    for (int i = 0; i < (bytesCount); ++i) { //De-interleave R & I
          r_value_f[i] = fft_out_f[(i * 2) + 0];
          i_value_f[i] = fft_out_f[(i * 2) + 1];
    }

    test_vec_atan2f(output_ang, i_value_f, r_value_f, bytesCount);

    // Quantize the feature vector data using the model's scale and zero-point
    float input_scale = input->params.scale;
    float input_scale_factor = 1/input_scale;
    int32_t input_zero_point = input->params.zero_point;

    for (int i = 0; i < bytesCount; ++i)
    {
#if PROBE_DATA_TO_INVOKE
        input->data.int8[i] = inject_in[sample_index];
        sample_index++;
#else
        float floatValueNorm =  (output_mag[i] * input_scale_factor) + input_zero_point;
        int32_t dwordInput = static_cast<int32_t>(roundf(floatValueNorm));
		CLAMP_VAL(dwordInput, INT8_MIN, INT8_MAX);
        int8_t byteInput = static_cast<int8_t>(dwordInput);
   		input->data.int8[i] = byteInput;
#endif
    }
    // Run the model on this input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

    // Get the output from the model, and make sure it's the expected size and
    // type.
    TF_LITE_MICRO_EXPECT_EQ(3, output->dims->size);
    TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
    TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output->type);

    int32_t output_zero_point = output->params.zero_point;
    float output_scale = output->params.scale;
    for (int i = 0; i < bytesCount; ++i)
    {
        COMPARE_VAL(output->data.int8[i], ref_out_NS[sample_out_index_cnt], is_mismatch_occured, sample_out_index_cnt, frame_num);
        model_output_f[i] = output_scale * (output->data.int8[i] - output_zero_point);
        predicted_mag_f[i] = model_output_f[i] * output_mag[i];
    }

    test_vec_cosinef(real_f, output_ang, bytesCount);
    test_vec_sinef(imag_f, output_ang, bytesCount);

    for(int i = 0; i < bytesCount; ++i)
    {
        ifft_input[2*i] = real_f[i] * predicted_mag_f[i];
        ifft_input[2*i+1] = imag_f[i] * predicted_mag_f[i];
    }

    test_ifft_real_512(ifft_output, ifft_input, bytesCount);

    //Final output -- overlap-add
    for(int i = STRIDE_LENGTH;i < (STRIDE_LENGTH+FRAME_LENGTH); i++)
            overlap_add_buf[i] = overlap_add_buf[i] + ifft_output[i-(STRIDE_LENGTH)];
    for(int i = 0;i < FRAME_LENGTH; i++)
            overlap_add_buf[i] = overlap_add_buf[i+(STRIDE_LENGTH)];
    for(int i = FRAME_LENGTH;i < (STRIDE_LENGTH+FRAME_LENGTH); i++)
            overlap_add_buf[i] = 0;
    for(int i = 0; i < STRIDE_LENGTH; i++) {
        final_output[i] = overlap_add_buf[i];
        final_output[i] = (float)((final_output[i]*Inv_sqrt_2_q15)/32768);
        //final_output[i] = (float)(final_output[i]/2);
    }

    FloatToFixed_Q1_15_Sat_vec(final_output, output_int16, STRIDE_LENGTH);

    fwrite(output_int16, sizeof(int16_t), (size_t)STRIDE_LENGTH, fout);

    printf("frame #%d\n", frame_num);
    frame_num++;
    }

	COMPARE_RESULT(is_mismatch_occured);
    FILE_CLOSE(fp);
    FILE_CLOSE(fout);
#if PROBE_DATA_TO_INVOKE
    FREE(inject_in);
#endif
#if COMPARISON_ENABLED
    FREE(ref_out_NS);
#endif
  }
  else
  {
  /* Process one line at a time */
  while(fgets(curr_cmd, MAX_CMD_LINE_LENGTH, param_file_id))
  {
      curpos = 0;
      fargc = 0;
      /* if it is not a param_file command and if */
      /* CLP processing is not enabled */
      if(curr_cmd[0] != '@' && !processcmd)     /* skip it */
      {
          continue;
      }
      // Reserver 0 for the binary name
      strncpy(fargv[0], argv[0], MAX_CMD_LINE_LENGTH);
      fargc++;

      while(sscanf(curr_cmd + curpos, "%s", fargv[fargc]) != EOF)
      {
          if(fargv[0][0]=='/' && fargv[0][1]=='/')
              break;
          if(strcmp(fargv[0], "@echo") == 0)
              break;
          if(strcmp(fargv[fargc], "@New_line") == 0)
          {
              char * str = fgets(curr_cmd + curpos, MAX_CMD_LINE_LENGTH,
                      param_file_id);
              (void)str;
              continue;
          }
          curpos += strlen(fargv[fargc]);
          while(*(curr_cmd + curpos)==' ' || *(curr_cmd + curpos)=='\t')
              curpos++;
          fargc++;
      }

      if(fargc < 2) /* for blank lines etc. */
          continue;

      if(strcmp(fargv[1], "@Output_path") == 0)
      {
          if(fargc > 2) strcpy(pb_output_file_path, fargv[2]);
          else strcpy(pb_output_file_path, "output.raw");
          continue;
      }

      if(strcmp(fargv[1], "@Input_path") == 0)
      {
          if(fargc > 2) strcpy(pb_input_file_path, fargv[2]);
          else strcpy(pb_input_file_path, "");
          continue;
      }

      if(strcmp(fargv[1], "@Start") == 0)
      {
          processcmd = 1;
          continue;
      }

      if(strcmp(fargv[1], "@Stop") == 0)
      {
          processcmd = 0;
          continue;
      }

      /* otherwise if this a normal command and its enabled for execution */
      if(processcmd)
      {
          FILE *fp, *fout;
          if(fargc<3)
          {
              printf("Usage: %s <input-raw-file.raw> <test_result>\n", argv[0]);
              exit(0);
          }
          strncpy(pb_input_file_name, fargv[1],MAX_CMD_LINE_LENGTH);
          prefix_inpdir_name(pb_input_file_name);
          FILE_OPEN(fp, pb_input_file_name, "rb");
          strncpy(pb_output_file_name, fargv[2],MAX_CMD_LINE_LENGTH);
          prefix_outdir_name(pb_output_file_name);
          FILE_OPEN(fout, pb_output_file_name, "wb");

          printf("Processing input file: '%s'\n", pb_input_file_name);
          fseek(fp, 0, SEEK_END);
          long lSize = ftell (fp) - WAV_HEADER_SIZE; //find total samples
          rewind (fp);
          //int num_samples = (int)floor(lSize/sizeof(int16_t));
      	  int total_frames = (int)floor(lSize/(STRIDE_LENGTH*sizeof(int16_t)));

    	  write_PCM16_mono_header(fout, SAMPLE_RATE, total_frames*STRIDE_LENGTH);
          frame_num = 0;

          interpreter.ResetVariableTensors();

#if PROBE_DATA_TO_INVOKE
          pb_input_file_name[strlen(pb_input_file_name)-4] = '\0';
          printf("Injecting Invoke1 input from Python Ref...\n");
          static int sample_index = 0;
          int neg = 0;
          char ch;
          sample_index = 0;
          FILE *finject;
          FILE_OPEN(finject, strcat(pb_input_file_name,"_invoke1_input.txt"),"r");
          MEM_ALLOC(inject_in, total_frames*bytesCount, short);
          memset(inject_in,  0, total_frames*bytesCount*sizeof(short));
          do {
              ch = fgetc(finject);
              if( feof(finject) ) {
                 break ;
              }
              if(ch == ',')
              {
            	  if(neg == 1)
            		  inject_in[sample_index] *= -1;
            	  sample_index++;
            	  neg = 0;
              }
              else if(ch == '-')
            	  neg = 1;
              else
            	  inject_in[sample_index] = inject_in[sample_index]*10 + (short)(ch - '0');
           } while(1);
          FILE_CLOSE(finject);
          printf("Injecting Done.\n");
          sample_index = 0;
#endif //PROBE_DATA_TO_INVOKE
#if COMPARISON_ENABLED
          pb_input_file_name[strlen(pb_input_file_name)-strlen("_invoke1_input.txt")] = '\0';
          static int sample_out_index_cnt = 0;
          neg = 0;
          sample_out_index_cnt = 0;
          FILE *frefout;
          FILE_OPEN(frefout, strcat(pb_input_file_name,"_invoke1_output.txt"), "r");
          MEM_ALLOC(ref_out_NS, total_frames*bytesCount, short);
          memset(ref_out_NS,  0, total_frames*bytesCount*sizeof(short));
          do {
              ch = fgetc(frefout);
              if( feof(frefout) ) {
                 break ;
              }
              if(ch == ',')
              {
            	  if(neg == 1)
            		  ref_out_NS[sample_out_index_cnt] *= -1;
            	  sample_out_index_cnt++;
            	  neg = 0;
              }
              else if(ch == '-')
            	  neg = 1;
              else
            	  ref_out_NS[sample_out_index_cnt] = ref_out_NS[sample_out_index_cnt]*10 + (short)(ch - '0');
           } while(1);
          FILE_CLOSE(frefout);
          sample_out_index_cnt = 0;
#endif
          /*  Main While loop */
      	for(int frm=0; frm<total_frames; frm++)
      	{
          fseek(fp, WAV_HEADER_SIZE+(frm*STRIDE_LENGTH*sizeof(int16_t)), SEEK_SET);
          int read_samples = fread(preinput, sizeof(int16_t), (size_t)FRAME_LENGTH, fp);
      	if(read_samples < FRAME_LENGTH)
      	{
#if PROBE_DATA_TO_INVOKE
            break;
#endif
      		for(int i=read_samples; i<FRAME_LENGTH; i++)
      			preinput[i] = 0;
      	}
		// Pre-processing code (STFT)
        test_fft_real_512(preinput, fft_out_f, FRAME_LENGTH);

		complex_mag cmolx_fourier;
		cmolx_fourier.mixed = fft_out_f;
		test_vec_complex2mag(output_mag, cmolx_fourier.complex, bytesCount);

		for (int i = 0; i < (bytesCount); ++i) { //De-interleave R & I
			  r_value_f[i] = fft_out_f[(i * 2) + 0];
			  i_value_f[i] = fft_out_f[(i * 2) + 1];
		}
        
		test_vec_atan2f(output_ang, i_value_f, r_value_f, bytesCount);

		// Quantize the feature vector data using the model's scale and zero-point
		float input_scale = input->params.scale;
		float input_scale_factor = 1/input_scale;
		int32_t input_zero_point = input->params.zero_point;

		for (int i = 0; i < bytesCount; ++i)
		{
#if PROBE_DATA_TO_INVOKE
			input->data.int8[i] = inject_in[sample_index];
			sample_index++;
#else
			float floatValueNorm =  (output_mag[i] * input_scale_factor) + input_zero_point;
			int32_t dwordInput = static_cast<int32_t>(roundf(floatValueNorm));
			CLAMP_VAL(dwordInput, INT8_MIN, INT8_MAX);
			int8_t byteInput = static_cast<int8_t>(dwordInput);
			input->data.int8[i] = byteInput;
#endif
		}
		// Run the model on this input and make sure it succeeds.
		TfLiteStatus invoke_status = interpreter.Invoke();
		if (invoke_status != kTfLiteOk) {
		  TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
		}
		TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

		// Get the output from the model, and make sure it's the expected size and
		// type.
		TF_LITE_MICRO_EXPECT_EQ(3, output->dims->size);
		TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
		TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
		TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output->type);

		int32_t output_zero_point = output->params.zero_point;
		float output_scale = output->params.scale;
		for (int i = 0; i < bytesCount; ++i)
		{
			COMPARE_VAL(output->data.int8[i], ref_out_NS[sample_out_index_cnt], is_mismatch_occured, sample_out_index_cnt, frame_num);
			model_output_f[i] = output_scale * (output->data.int8[i] - output_zero_point);
			predicted_mag_f[i] = model_output_f[i] * output_mag[i];
		}

		test_vec_cosinef(real_f,output_ang,bytesCount);
		test_vec_sinef(imag_f,output_ang,bytesCount);

		for(int i = 0; i < bytesCount; ++i)
		{
			ifft_input[2*i] = real_f[i] * predicted_mag_f[i];
			ifft_input[2*i+1] = imag_f[i] * predicted_mag_f[i];
		}

        test_ifft_real_512(ifft_output, ifft_input, bytesCount);

		//Final output -- overlap-add
		for(int i = STRIDE_LENGTH;i < (STRIDE_LENGTH+FRAME_LENGTH); i++)
				overlap_add_buf[i] = overlap_add_buf[i] + ifft_output[i-(STRIDE_LENGTH)];
		for(int i = 0;i < FRAME_LENGTH; i++)
				overlap_add_buf[i] = overlap_add_buf[i+(STRIDE_LENGTH)];
		for(int i = FRAME_LENGTH;i < (STRIDE_LENGTH+FRAME_LENGTH); i++)
				overlap_add_buf[i] = 0;
		for(int i = 0; i < STRIDE_LENGTH; i++) {
			final_output[i] = overlap_add_buf[i];
			final_output[i] = (float)((final_output[i]*Inv_sqrt_2_q15)/32768);
		}

        FloatToFixed_Q1_15_Sat_vec(final_output, output_int16, STRIDE_LENGTH);

        fwrite(output_int16, sizeof(int16_t), (size_t)STRIDE_LENGTH, fout);

        printf("frame #%d\n", frame_num);
        frame_num++;
        }
	  
    	COMPARE_RESULT(is_mismatch_occured);

        FILE_CLOSE(fp);
        FILE_CLOSE(fout);
#if PROBE_DATA_TO_INVOKE
           FREE(inject_in);
#endif
#if COMPARISON_ENABLED
           FREE(ref_out_NS);
#endif
        }
    }
  }

  FREE(fft_out_f);
  FREE(overlap_add_buf);
  FREE(preinput);
  FREE(preinput_f);
  FREE(output_mag);
  FREE(r_value_f);
  FREE(i_value_f);
  FREE(output_ang);
  FREE(model_output_f);
  FREE(predicted_mag_f);
  FREE(real_f);
  FREE(imag_f);
  FREE(ifft_input);
  FREE(ifft_output);
  FREE(final_output);
  FREE(output_int16);
  TF_LITE_REPORT_ERROR(&micro_error_reporter, "Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
