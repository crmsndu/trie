# Vendored SGLang base and strict-replay integration

The `sglang/` directory was imported from the exact PVC checkout used for the
GLM-5.2 CP/P-D/MTP experiments, then extended with the completion token-ID
response needed by Trie's strict replay mode.

- Upstream: `https://github.com/sgl-project/sglang.git`
- PVC checkout: `sglang-glm52-active-index-kv`
- Commit: `e701218238227b7bf2737a4b1090fa6a8534c5ec`
- Tree: `5095967943c955e27fa913c8e36c87713a500b2b`
- Base PR commit: `db39fbd9fb5df3e5e1705d4fb6807d29498a4488`
  (`sgl-project/sglang#29847`, DSA CP shared KV cache)

The snapshot also includes the local GLM-5.2 patch chain for MTP IndexShare
stability, prefill-CP index sharing, CP-shared P/D transfer, FP8/VMM capacity
accounting, active-index KV materialization, and compact DSA state transfer.

Before the strict-replay edits, the source and imported trees were verified
byte-for-byte by comparing their Git tree object IDs. The integration changes
are limited to the OpenAI completion protocol/serving path and its unit tests.
