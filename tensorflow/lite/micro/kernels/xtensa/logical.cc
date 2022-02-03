/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/kernels/logical.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {
namespace {

#if defined(HIFI4) || defined(HIFI5)
TfLiteStatus LogicalImpl_HiFi(TfLiteContext* context, TfLiteNode* node,
                         bool (*func)(bool, bool)) {
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kLogicalInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kLogicalInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kLogicalOutputTensor);

  if (tflite::micro::HaveSameShapes(input1, input2)) {
    int err;
    const int8_t *input1_data_ptr, *input2_data_ptr;
    int8_t *output_data_ptr;
    RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
    RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
    RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
    const int flat_size = MatchingFlatSize(input1_shape, input2_shape, output_shape);

    input1_data_ptr  = tflite::micro::GetTensorData<int8_t>(input1);
    input2_data_ptr  = tflite::micro::GetTensorData<int8_t>(input2);
    output_data_ptr  = tflite::micro::GetTensorData<int8_t>(output);

    if (func == LogicalAnd) {
      err = xa_nn_elm_logicaland_boolxbool_bool(
          output_data_ptr,
          input1_data_ptr,
          input2_data_ptr,
          flat_size);
      TF_LITE_ENSURE(context, err == 0);
    } else if (func == LogicalOr) {
      err = xa_nn_elm_logicalor_boolxbool_bool(
          output_data_ptr,
          input1_data_ptr,
          input2_data_ptr,
          flat_size);
      TF_LITE_ENSURE(context, err == 0);
    } else {
      err = 1;
      TF_LITE_ENSURE(context, err == 0);
    }
  } else {
    reference_ops::BroadcastBinaryFunction4DSlow<bool, bool, bool>(
        tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<bool>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<bool>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<bool>(output), func);
  }

  return kTfLiteOk;
}
#endif // defined(HIFI4) || defined(HIFI5)

TfLiteStatus LogicalOrEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4) || defined(HIFI5)
  return LogicalImpl_HiFi(context, node, LogicalOr);
#else
  return LogicalImpl(context, node, LogicalOr);
#endif
}

TfLiteStatus LogicalAndEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4) || defined(HIFI5)
  return LogicalImpl_HiFi(context, node, LogicalAnd);
#else
  return LogicalImpl(context, node, LogicalAnd);
#endif
}

}  // namespace

TfLiteRegistration Register_LOGICAL_OR() {
  // Init, Free, Prepare, Eval are satisfying the Interface required by
  // TfLiteRegistration.
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/LogicalOrEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_LOGICAL_AND() {
  // Init, Free, Prepare, Eval are satisfying the Interface required by
  // TfLiteRegistration.
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/LogicalAndEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
