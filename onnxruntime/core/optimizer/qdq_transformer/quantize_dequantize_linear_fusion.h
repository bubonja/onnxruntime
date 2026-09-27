// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
    @Class QuantizeDequantizeLinearFusion

    Replaces a 16-bit QuantizeLinear whose consumers are all DequantizeLinear nodes with the same constant scalar
    scale and zero point by a single com.microsoft.QuantizeDequantizeLinear node. The result is bitwise identical,
    but the quantized tensor and the extra nodes and kernel launches are removed.

    It must run after the transformers that fuse Q/DQ node units, since it removes the DQ/Q nodes they match.
*/
class QuantizeDequantizeLinearFusion : public GraphTransformer {
 public:
  explicit QuantizeDequantizeLinearFusion(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("QuantizeDequantizeLinearFusion", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
