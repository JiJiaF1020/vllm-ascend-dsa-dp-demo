# Design Documents

This section provides an overview of the features implemented in vLLM Ascend. Developers can refer to this guide to understand how vLLM Ascend works.

The DSA sparse-offload documentation is collected in
{doc}`dsa_sparse_offload/index`, separating system architecture, feature specifications,
the feature roadmap, design proposals, and engineering diagnostics.

:::{toctree}
:caption: Design Documents
:maxdepth: 1
patch
cpu_binding
ModelRunner_prepare_inputs
disaggregated_prefill
eplb_swift_balancer
ACL_Graph
KV_Cache_Pool_Guide
dsa_sparse_offload/index
add_custom_aclnn_op
context_parallel
dynamic_chunked_pipeline_parallel
quantization
npugraph_ex
:::
