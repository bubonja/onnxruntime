// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>

#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

// Bitwise identical to the CPU QuantizeLinear kernel followed by the CPU DequantizeLinear kernel.
template <typename T>
class QuantizeDequantizeLinear final : public OpKernel {
 public:
  explicit QuantizeDequantizeLinear(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
Status QuantizeDequantizeLinear<T>::Compute(OpKernelContext* context) const {
  const Tensor& x = *context->Input<Tensor>(0);
  const Tensor& y_scale = *context->Input<Tensor>(1);
  const Tensor& y_zero_point = *context->Input<Tensor>(2);
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(&y_scale) && IsScalarOr1ElementVector(&y_zero_point),
                    "QuantizeDequantizeLinear requires a scalar y_scale and y_zero_point.");

  Tensor& y = *context->Output(0, x.Shape());
  const std::ptrdiff_t count = narrow<std::ptrdiff_t>(x.Shape().Size());
  if (count == 0) {
    return Status::OK();
  }

  const float* input = x.Data<float>();
  float* output = y.MutableData<float>();
  const float scale = *y_scale.Data<float>();
  const T zero_point = *y_zero_point.Data<T>();

  // Same partitioning as ParQuantizeLinearStd: every MLAS call starts on a 128-element boundary, so each element
  // takes the same vector or tail path as in QuantizeLinear.
  constexpr std::ptrdiff_t kBlockSize = 128;
  constexpr std::ptrdiff_t kChunkSize = 8 * kBlockSize;
  const std::ptrdiff_t num_blocks = (count + kBlockSize - 1) / kBlockSize;
  const TensorOpCost unit_cost{static_cast<double>(kBlockSize * sizeof(float)),
                               static_cast<double>(kBlockSize * sizeof(float)),
                               static_cast<double>(kBlockSize) * 3.0};
  concurrency::ThreadPool::TryParallelFor(
      context->GetOperatorThreadPool(), num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        // Local copies: the output stores could otherwise alias the captured references and block vectorization.
        const float scale_value = scale;
        const int32_t zero_point_int = static_cast<int32_t>(zero_point);
        T quantized[kChunkSize];
        const std::ptrdiff_t stop = std::min(count, end * kBlockSize);
        for (std::ptrdiff_t start = begin * kBlockSize; start < stop; start += kChunkSize) {
          const std::ptrdiff_t n = std::min(kChunkSize, stop - start);
          MlasQuantizeLinear(input + start, quantized, static_cast<size_t>(n), scale_value, zero_point);
          // Same expression as DequantizeLinear.
          float* chunk_output = output + start;
          for (std::ptrdiff_t i = 0; i < n; ++i) {
            chunk_output[i] = static_cast<float>(static_cast<int32_t>(quantized[i]) - zero_point_int) * scale_value;
          }
        }
      });

  return Status::OK();
}

#define REGISTER_QUANTIZE_DEQUANTIZE_LINEAR(T)                        \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                  \
      QuantizeDequantizeLinear,                                       \
      1,                                                              \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeDequantizeLinear<T>);

REGISTER_QUANTIZE_DEQUANTIZE_LINEAR(uint16_t)
REGISTER_QUANTIZE_DEQUANTIZE_LINEAR(int16_t)

}  // namespace contrib
}  // namespace onnxruntime
