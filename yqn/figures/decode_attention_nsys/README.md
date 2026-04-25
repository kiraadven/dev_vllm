# Decode Attention Nsight Analysis

- This tool only analyzes NVTX ranges whose label starts with `decode_attn`.
- If a report has no NVTX table or no `decode_attn` ranges, request-level attention timing cannot be reconstructed from that report.

## Reports

- `yqn/disaggregated_serving_dp/logs_dp_2p2d_nixl/decode_attn_nsys/decode1.nsys-rep` -> `yqn/figures/decode_attention_nsys/yqn__disaggregated_serving_dp__logs_dp_2p2d_nixl__decode_attn_nsys__decode1/summary.md`
- `yqn/disaggregated_serving_dp/logs_dp_2p2d_nixl/decode_attn_nsys/decode2.nsys-rep` -> `yqn/figures/decode_attention_nsys/yqn__disaggregated_serving_dp__logs_dp_2p2d_nixl__decode_attn_nsys__decode2/summary.md`
