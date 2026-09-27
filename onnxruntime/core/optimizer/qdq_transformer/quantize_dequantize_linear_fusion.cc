// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/quantize_dequantize_linear_fusion.h"

#include "core/graph/constants.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {
namespace {

int32_t ElemType(const NodeArg& arg) {
  const auto* type = arg.TypeAsProto();
  return type != nullptr && type->has_tensor_type() ? type->tensor_type().elem_type()
                                                    : ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
}

bool HasNoBlockSize(const Node& node) {
  const auto* block_size = graph_utils::GetNodeAttribute(node, "block_size");
  return block_size == nullptr || block_size->i() == 0;
}

bool IsFusableQ(const Graph& graph, const Node& q_node, const QDQ::GetConstantInitializerFn& get_constant_initializer,
                const InlinedHashSet<std::string_view>& compatible_eps) {
  if (!QDQ::MatchQNode(q_node) || !graph_utils::IsSupportedProvider(q_node, compatible_eps) ||
      !HasNoBlockSize(q_node) || q_node.GetOutputEdgesCount() == 0 || graph.NodeProducesGraphOutput(q_node)) {
    return false;
  }

  const auto& inputs = q_node.InputDefs();
  if (inputs.size() != QDQ::InputIndex::TOTAL_COUNT ||
      ElemType(*inputs[QDQ::InputIndex::INPUT_ID]) != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      ElemType(*inputs[QDQ::InputIndex::SCALE_ID]) != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return false;
  }

  const int32_t quantized_type = ElemType(*inputs[QDQ::InputIndex::ZERO_POINT_ID]);
  if (quantized_type != ONNX_NAMESPACE::TensorProto_DataType_UINT16 &&
      quantized_type != ONNX_NAMESPACE::TensorProto_DataType_INT16) {
    return false;
  }

  // A positive scale that compares equal to the DQ scale is also bitwise equal to it.
  return QDQ::IsQOrDQScalePositiveConstantScalar(graph, q_node, get_constant_initializer, graph.ModelPath());
}

// Returns the DQ consumers of q_node if every consumer is a DQ with the same scale and zero point. The DQ that
// produces a graph output, if any, is placed first.
InlinedVector<Node*> GetFusableDQs(Graph& graph, const Node& q_node,
                                   const QDQ::GetConstantInitializerFn& get_constant_initializer) {
  InlinedVector<Node*> dq_nodes;
  size_t graph_output_count = 0;
  for (auto it = q_node.OutputEdgesBegin(), end = q_node.OutputEdgesEnd(); it != end; ++it) {
    Node* dq_node = graph.GetNode(it->GetNode().Index());
    if (it->GetDstArgIndex() != QDQ::InputIndex::INPUT_ID || !QDQ::MatchDQNode(*dq_node) ||
        dq_node->GetExecutionProviderType() != q_node.GetExecutionProviderType() || !HasNoBlockSize(*dq_node) ||
        ElemType(*dq_node->OutputDefs()[0]) != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
        !QDQ::IsQDQPairSupported(graph, q_node, *dq_node, get_constant_initializer, graph.ModelPath(), false)) {
      return {};
    }

    if (graph.NodeProducesGraphOutput(*dq_node)) {
      if (++graph_output_count > 1) {
        return {};
      }
      dq_nodes.insert(dq_nodes.begin(), dq_node);
    } else {
      dq_nodes.push_back(dq_node);
    }
  }

  return dq_nodes;
}

}  // namespace

Status QuantizeDequantizeLinearFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                 const logging::Logger& logger) const {
  const auto get_constant_initializer = [&graph](const std::string& name) {
    return graph.GetConstantInitializer(name, true);
  };

  GraphViewer graph_viewer(graph);
  for (NodeIndex node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    Node* q_node = graph.GetNode(node_index);
    if (q_node == nullptr) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*q_node, modified, graph_level, logger));

    if (!IsFusableQ(graph, *q_node, get_constant_initializer, GetCompatibleExecutionProviders())) {
      continue;
    }

    const InlinedVector<Node*> dq_nodes = GetFusableDQs(graph, *q_node, get_constant_initializer);
    if (dq_nodes.empty()) {
      continue;
    }

    // The fused node produces the first DQ's output (the graph output, if any); the other DQs' consumers are
    // rewired to it.
    auto& q_inputs = q_node->MutableInputDefs();
    Node& fused = graph.AddNode(graph.GenerateNodeName(q_node->Name() + "_QuantizeDequantizeLinear"),
                                "QuantizeDequantizeLinear", "Fused QuantizeLinear and DequantizeLinear",
                                {q_inputs[0], q_inputs[1], q_inputs[2]}, {dq_nodes.front()->MutableOutputDefs()[0]},
                                nullptr, kMSDomain);
    fused.SetExecutionProviderType(q_node->GetExecutionProviderType());

    InlinedVector<NodeIndex> source_nodes;
    source_nodes.reserve(dq_nodes.size() + 1);
    source_nodes.push_back(q_node->Index());

    graph_utils::MoveAllNodeInputEdges(graph, *q_node, fused);
    for (Node* dq_node : dq_nodes) {
      graph_utils::ReplaceDownstreamNodeInput(graph, *dq_node, 0, fused, 0);
      source_nodes.push_back(dq_node->Index());
    }

    graph_utils::RemoveNodeOutputEdges(graph, *q_node);
    for (NodeIndex index : source_nodes) {
      graph.RemoveNode(index);
    }
    graph.NotifyNodeReplacement(source_nodes, fused.Index());

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
