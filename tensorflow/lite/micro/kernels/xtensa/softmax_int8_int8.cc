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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_softmax.h"

namespace tflite {
namespace {

#if defined(FUSION_F1) || defined(HIFI5)
TfLiteStatus EvalHifiInt8(const XtensaSoftmaxOpData* op_data,
                      const TfLiteEvalTensor* input, TfLiteEvalTensor* output,
                      TfLiteContext* context) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  void* p_scratch = static_cast<void*>(
      context->GetScratchBuffer(context, op_data->scratch_tensor_index));
  for (int i = 0; i < outer_size; ++i) {
    int err = xa_nn_vec_softmax_asym8s_asym8s(
        &output_data[i * depth], &input_data[i * depth],
        op_data->params.diff_min, op_data->params.input_left_shift,
        op_data->params.input_multiplier, depth, p_scratch);
    TF_LITE_ENSURE(context, err == 0);
  }
  return kTfLiteOk;
}
#endif  // defined(FUSION_F1) || defined(HIFI5)

}  // namespace

TfLiteStatus XtensaEvalSoftmaxInt8Int8(TfLiteContext* context,
                                        TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TFLITE_DCHECK(node->user_data != nullptr);

  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
#if defined(FUSION_F1) || defined(HIFI5)
    return EvalHifiInt8(static_cast<XtensaSoftmaxOpData*>(node->user_data), input,
                    output, context);
#else
    SoftmaxParams op_data = *static_cast<SoftmaxParams*>(node->user_data);
    tflite::reference_ops::Softmax(
        op_data, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
    return kTfLiteOk;
#endif
  } else {
    TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                       TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }
}

}  // namespace tflite
