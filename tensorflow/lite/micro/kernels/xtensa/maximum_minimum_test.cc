/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void TestMaxMinFloat(const TfLiteRegistration& registration,
                     int* input1_dims_data, const float* input1_data,
                     int* input2_dims_data, const float* input2_data,
                     const float* expected_output_data, int* output_dims_data,
                     float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

template <typename data_type>
void TestMaxMinQuantized(const TfLiteRegistration& registration,
                         int* input1_dims_data, const data_type* input1_data,
                         const float input1_scale, const int input1_zero_point,
                         int* input2_dims_data, const data_type* input2_data,
                         const float input2_scale, const int input2_zero_point,
                         const data_type* expected_output_data,
                         const float output_scale, const int output_zero_point,
                         int* output_dims_data, data_type* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elm_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, input1_scale,
                            input1_zero_point),
      CreateQuantizedTensor(input2_data, input2_dims, input2_scale,
                            input2_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  if(expected_output_data != NULL){                 // skip check for NULL reference
    //printf("Comparing reference with obtained results\n");
    for (int i = 0; i < output_elm_count; ++i) {
      TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
    }
  }
}


}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatTest) {
  int dims[] = {3, 3, 1, 2};
  const float data1[] = {1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  const float data2[] = {-1.0, 0.0, 1.0, 12.0, -3.0, -1.43};
  const float golden_max[] = {1.0, 0.0, 1.0, 12.0, -2.0, -1.43};
  const float golden_min[] = {-1.0, 0.0, -1.0, 11.0, -3.0, -1.44};
  float output_data[6];

  tflite::testing::TestMaxMinFloat(tflite::ops::micro::Register_MAXIMUM(), dims,
                                   data1, dims, data2, golden_max, dims,
                                   output_data);

  tflite::testing::TestMaxMinFloat(tflite::ops::micro::Register_MINIMUM(), dims,
                                   data1, dims, data2, golden_min, dims,
                                   output_data);
}

TF_LITE_MICRO_TEST(FloatWithBroadcastTest) {
  int dims[] = {3, 3, 1, 2};
  int dims_scalar[] = {1, 2};
  const float data1[] = {1.0, 0.0, -1.0, -2.0, -1.44, 11.0};
  const float data2[] = {0.5, 2.0};
  const float golden_max[] = {1.0, 2.0, 0.5, 2.0, 0.5, 11.0};
  const float golden_min[] = {0.5, 0.0, -1.0, -2.0, -1.44, 2.0};
  float output_data[6];

  tflite::testing::TestMaxMinFloat(tflite::ops::micro::Register_MAXIMUM(), dims,
                                   data1, dims_scalar, data2, golden_max, dims,
                                   output_data);

  tflite::testing::TestMaxMinFloat(tflite::ops::micro::Register_MINIMUM(), dims,
                                   data1, dims_scalar, data2, golden_min, dims,
                                   output_data);
}

TF_LITE_MICRO_TEST(Int8) {
  int input1_dims[] = {4, 1, 8, 1, 2};
  int input2_dims[] = {4, 1, 8, 1, 2};
  int    res_dims[] = {4, 1, 8, 1, 2};

  const int8_t input1_data[] = 
    {  -74,  65,  -99, -85, -74,  19,  -99, -85, -74,  76,  -99, -85,  65,  65,   -1, -84};

  const int8_t input2_data[] =
    {   65,  76,   65,  65,  19,  76,   19,  19,  88,  88,   88,  88, 109, 125,   65,  65};

  const int8_t golden_max[] = 
    {   65,  76,   65,  65,  19,  76,   19,  19,  88,  88,   88,  88, 109, 125,   65,  65};

  const int8_t golden_min[] =
    {  -74,  65,  -99, -85, -74,  19,  -99, -85, -74,  76,  -99, -85,  65,  65,   -1, -84};


  // +1 in case you need to check for unaligned buffer
  __attribute__ ((aligned(32))) int8_t out_buffer[sizeof(golden_max) + 1];
  
  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
      
  tflite::testing::TestMaxMinQuantized(
      tflite::ops::micro::Register_MAXIMUM(),
      input1_dims, input1_data, input_scale, input_zero_point,
      input2_dims, input2_data, input_scale, input_zero_point,
      golden_max, output_scale, output_zero_point, res_dims, out_buffer);

  tflite::testing::TestMaxMinQuantized(
      tflite::ops::micro::Register_MINIMUM(),
      input1_dims, input1_data, input_scale, input_zero_point,
      input2_dims, input2_data, input_scale, input_zero_point,
      golden_min, output_scale, output_zero_point, res_dims, out_buffer);
}

TF_LITE_MICRO_TEST(Int8Broadcast4D_BothBroadcast) {
  int input1_dims[] = {4, 2, 1, 3, 1};
  int input2_dims[] = {3,    5, 1, 4};
  int    res_dims[] = {4, 2, 5, 3, 4};

  const int8_t input1_data[] = 
    {   65,  19,  88,  93,   -3, 125};

  const int8_t input2_data[] =
    {  -74,  76,  -99, -85, 109, 125,   -1, -84,  71,  -6,   17, -84, 112,   2, -107,  35,
        75, 116,  126, -86 };

  const int8_t golden_max[] = 
    {   65,  76,   65,  65,  19,  76,   19,  19,  88,  88,   88,  88, 109, 125,   65,  65,
       109, 125,   19,  19, 109, 125,   88,  88,  71,  65,   65,  65,  71,  19,   19,  19,
        88,  88,   88,  88, 112,  65,   65,  65, 112,  19,   19,  35, 112,  88,   88,  88,
        75, 116,  126,  65,  75, 116,  126,  19,  88, 116,  126,  88,  93,  93,   93,  93,
        -3,  76,   -3,  -3, 125, 125,  125, 125, 109, 125,   93,  93, 109, 125,   -1,  -3,
       125, 125,  125, 125,  93,  93,   93,  93,  71,  -3,   17,  -3, 125, 125,  125, 125,
       112,  93,   93,  93, 112,   2,   -3,  35, 125, 125,  125, 125,  93, 116,  126,  93,
        75, 116,  126,  -3, 125, 125,  126, 125 };

  const int8_t golden_min[] =
  {    -74,  65,  -99, -85, -74,  19,  -99, -85, -74,  76,  -99, -85,  65,  65,   -1, -84,
        19,  19,   -1, -84,  88,  88,   -1, -84,  65,  -6,   17, -84,  19,  -6,   17, -84,
        71,  -6,   17, -84,  65,   2, -107,  35,  19,   2, -107,  19,  88,   2, -107,  35,
        65,  65,   65, -86,  19,  19,   19, -86,  75,  88,   88, -86, -74,  76,  -99, -85,
       -74,  -3,  -99, -85, -74,  76,  -99, -85,  93,  93,   -1, -84,  -3,  -3,   -3, -84,
       109, 125,   -1, -84,  71,  -6,   17, -84,  -3,  -6,   -3, -84,  71,  -6,   17, -84,
        93,   2, -107,  35,  -3,  -3, -107,  -3, 112,   2, -107,  35,  75,  93,   93, -86,
        -3,  -3,   -3, -86,  75, 116,  125, -86 };


  // +1 in case you need to check for unaligned buffer
  __attribute__ ((aligned(32))) int8_t out_buffer[sizeof(golden_max) + 1];
  
  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
      
  tflite::testing::TestMaxMinQuantized(
      tflite::ops::micro::Register_MAXIMUM(),
      input1_dims, input1_data, input_scale, input_zero_point,
      input2_dims, input2_data, input_scale, input_zero_point,
      golden_max, output_scale, output_zero_point, res_dims, out_buffer);

  tflite::testing::TestMaxMinQuantized(
      tflite::ops::micro::Register_MINIMUM(),
      input1_dims, input1_data, input_scale, input_zero_point,
      input2_dims, input2_data, input_scale, input_zero_point,
      golden_min, output_scale, output_zero_point, res_dims, out_buffer);
}

TF_LITE_MICRO_TEST(Int8Broadcast4D_T1Broadcast) {
  int input1_dims[] = {3,    5, 1, 4};
  int input2_dims[] = {4, 2, 5, 3, 4};
  int    res_dims[] = {4, 2, 5, 3, 4};

  const int8_t input1_data[] =
    {  -74,  76,  -99, -85, 109, 125,   -1, -84,  71,  -6,   17, -84, 112,   2, -107,  35,
        75, 116,  126, -86 };

  const int8_t input2_data[] = 
    {   65,  76,   65,  65,  19,  76,   19,  19,  88,  88,   88,  88, 109, 125,   65,  65,
       109, 125,   19,  19, 109, 125,   88,  88,  71,  65,   65,  65,  71,  19,   19,  19,
        88,  88,   88,  88, 112,  65,   65,  65, 112,  19,   19,  35, 112,  88,   88,  88,
        75, 116,  126,  65,  75, 116,  126,  19,  88, 116,  126,  88,  93,  93,   93,  93,
        -3,  76,   -3,  -3, 125, 125,  125, 125, 109, 125,   93,  93, 109, 125,   -1,  -3,
       125, 125,  125, 125,  93,  93,   93,  93,  71,  -3,   17,  -3, 125, 125,  125, 125,
       112,  93,   93,  93, 112,   2,   -3,  35, 125, 125,  125, 125,  93, 116,  126,  93,
        75, 116,  126,  -3, 125, 125,  126, 125 };

  const int8_t golden_max[] = 
    {   65,  76,   65,  65,  19,  76,   19,  19,  88,  88,   88,  88, 109, 125,   65,  65,
       109, 125,   19,  19, 109, 125,   88,  88,  71,  65,   65,  65,  71,  19,   19,  19,
        88,  88,   88,  88, 112,  65,   65,  65, 112,  19,   19,  35, 112,  88,   88,  88,
        75, 116,  126,  65,  75, 116,  126,  19,  88, 116,  126,  88,  93,  93,   93,  93,
        -3,  76,   -3,  -3, 125, 125,  125, 125, 109, 125,   93,  93, 109, 125,   -1,  -3,
       125, 125,  125, 125,  93,  93,   93,  93,  71,  -3,   17,  -3, 125, 125,  125, 125,
       112,  93,   93,  93, 112,   2,   -3,  35, 125, 125,  125, 125,  93, 116,  126,  93,
        75, 116,  126,  -3, 125, 125,  126, 125 };

  const int8_t golden_min[] =
  {    -74,  76,  -99, -85, -74,  76,  -99, -85, -74,  76,  -99, -85, 109, 125,   -1, -84,
       109, 125,   -1, -84, 109, 125,   -1, -84,  71,  -6,   17, -84,  71,  -6,   17, -84,
        71,  -6,   17, -84, 112,   2, -107,  35, 112,   2, -107,  35, 112,   2, -107,  35,
        75, 116,  126, -86,  75, 116,  126, -86,  75, 116,  126, -86, -74,  76,  -99, -85,
       -74, 76 ,  -99, -85, -74,  76,  -99, -85, 109, 125,   -1, -84, 109, 125,   -1, -84,
       109, 125,   -1, -84,  71,  -6,   17, -84,  71,  -6,   17, -84,  71,  -6,   17, -84,
       112,   2, -107,  35, 112,   2, -107,  35, 112,   2, -107,  35,  75, 116,  126, -86,
        75, 116,  126, -86,  75, 116,  126, -86 };


  // +1 in case you need to check for unaligned buffer
  __attribute__ ((aligned(32))) int8_t out_buffer[sizeof(golden_max) + 1];
  
  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
      
  tflite::testing::TestMaxMinQuantized(
      tflite::ops::micro::Register_MAXIMUM(),
      input1_dims, input1_data, input_scale, input_zero_point,
      input2_dims, input2_data, input_scale, input_zero_point,
      golden_max, output_scale, output_zero_point, res_dims, out_buffer);

  tflite::testing::TestMaxMinQuantized(
      tflite::ops::micro::Register_MINIMUM(),
      input1_dims, input1_data, input_scale, input_zero_point,
      input2_dims, input2_data, input_scale, input_zero_point,
      golden_min, output_scale, output_zero_point, res_dims, out_buffer);
}

TF_LITE_MICRO_TEST(Int8Broadcast4D_T2Broadcast) {
  int input1_dims[] = {4, 2, 5, 3, 4};
  int input2_dims[] = {3,    5, 1, 4};
  int    res_dims[] = {4, 2, 5, 3, 4};

  const int8_t input1_data[] =
    {  -74,  65,  -99, -85, -74,  19,  -99, -85, -74,  76,  -99, -85,  65,  65,   -1, -84,
        19,  19,   -1, -84,  88,  88,   -1, -84,  65,  -6,   17, -84,  19,  -6,   17, -84,
        71,  -6,   17, -84,  65,   2, -107,  35,  19,   2, -107,  19,  88,   2, -107,  35,
        65,  65,   65, -86,  19,  19,   19, -86,  75,  88,   88, -86, -74,  76,  -99, -85,
       -74,  -3,  -99, -85, -74,  76,  -99, -85,  93,  93,   -1, -84,  -3,  -3,   -3, -84,
       109, 125,   -1, -84,  71,  -6,   17, -84,  -3,  -6,   -3, -84,  71,  -6,   17, -84,
        93,   2, -107,  35,  -3,  -3, -107,  -3, 112,   2, -107,  35,  75,  93,   93, -86,
        -3,  -3,   -3, -86,  75, 116,  125, -86 };

  const int8_t input2_data[] = 
    {  -74,  76,  -99, -85, 109, 125,   -1, -84,  71,  -6,   17, -84, 112,   2, -107,  35,
        75, 116,  126, -86 };

  const int8_t golden_max[] = 
    {  -74,  76,  -99, -85, -74,  76,  -99, -85, -74,  76,  -99, -85, 109, 125,   -1, -84,
       109, 125,   -1, -84, 109, 125,   -1, -84,  71,  -6,   17, -84,  71,  -6,   17, -84,
        71,  -6,   17, -84, 112,   2, -107,  35, 112,   2, -107,  35, 112,   2, -107,  35,
        75, 116,  126, -86,  75, 116,  126, -86,  75, 116,  126, -86, -74,  76,  -99, -85,
       -74,  76,  -99, -85, -74,  76,  -99, -85, 109, 125,   -1, -84, 109, 125,   -1, -84,
       109, 125,   -1, -84,  71,  -6,   17, -84,  71,  -6,   17, -84,  71,  -6,   17, -84,
       112,   2, -107,  35, 112,   2, -107,  35, 112,   2, -107,  35,  75, 116,  126, -86,
        75, 116,  126, -86,  75, 116,  126, -86 };

  const int8_t golden_min[] =
    {  -74,  65,  -99, -85, -74,  19,  -99, -85, -74,  76,  -99, -85,  65,  65,   -1, -84,
        19,  19,   -1, -84,  88,  88,   -1, -84,  65,  -6,   17, -84,  19,  -6,   17, -84,
        71,  -6,   17, -84,  65,   2, -107,  35,  19,   2, -107,  19,  88,   2, -107,  35,
        65,  65,   65, -86,  19,  19,   19, -86,  75,  88,   88, -86, -74,  76,  -99, -85,
       -74,  -3,  -99, -85, -74,  76,  -99, -85,  93,  93,   -1, -84,  -3,  -3,   -3, -84,
       109, 125,   -1, -84,  71,  -6,   17, -84,  -3,  -6,   -3, -84,  71,  -6,   17, -84,
        93,   2, -107,  35,  -3,  -3, -107,  -3, 112,   2, -107,  35,  75,  93,   93, -86,
        -3,  -3,   -3, -86,  75, 116,  125, -86 };


  // +1 in case you need to check for unaligned buffer
  __attribute__ ((aligned(32))) int8_t out_buffer[sizeof(golden_max) + 1];
  
  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
      
  tflite::testing::TestMaxMinQuantized(
      tflite::ops::micro::Register_MAXIMUM(),
      input1_dims, input1_data, input_scale, input_zero_point,
      input2_dims, input2_data, input_scale, input_zero_point,
      golden_max, output_scale, output_zero_point, res_dims, out_buffer);

  tflite::testing::TestMaxMinQuantized(
      tflite::ops::micro::Register_MINIMUM(),
      input1_dims, input1_data, input_scale, input_zero_point,
      input2_dims, input2_data, input_scale, input_zero_point,
      golden_min, output_scale, output_zero_point, res_dims, out_buffer);
}

TF_LITE_MICRO_TESTS_END
