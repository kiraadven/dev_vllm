# Decode Attention Summary: `yqn/disaggregated_serving_dp/logs_dp_2p2d_nixl/decode_attn_nsys/decode2.nsys-rep`

- SQLite: `yqn/disaggregated_serving_dp/logs_dp_2p2d_nixl/decode_attn_nsys/decode2.sqlite`
- NVTX table: `missing`
- decode_attn events found: `0`

## Interpretation

- Each `decode_attn ...` NVTX range is a batched decode attention region for one layer.
- This script attributes each batched range equally across all requests listed in `req_keys`.
- `request_avg_attn_per_layer.svg` uses x=request and y=average attributed attention time per layer invocation.

## Notes

- This SQLite export has no NVTX table, so decode attention time cannot be recovered from this report.

## Generated Files

