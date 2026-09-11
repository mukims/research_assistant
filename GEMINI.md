# Project Guidelines & Invariants: Research Assistant

## Quality & Verification Rules
- **Test and Retest Invariant**: Follow the [Mandatory Two-Cycle Verification Protocol](.agents/rules/test_and_retest.md) on every code change and deployment. Always execute two complete test cycles (Cycle 1: Implement -> Deploy -> Test; Cycle 2: Refine -> Redeploy -> Retest on warm state).
- **Ollama CPU Resource Limits**: On GCE CPU VMs, keep `num_ctx: 4096` in `config.py` and avoid manual thread overrides (`num_thread: 16`), Flash Attention, or quantized KV caches to prevent thread oversubscription and CPU lockups.
- **Persistent Data Volume**: Keep all vector data, indexed files, and manifests located in `DATA_DIR` (`/mnt/disks/data` on VM). Never wipe persistent storage.
