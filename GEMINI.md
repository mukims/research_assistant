# Project Guidelines & Invariants: Research Assistant

## Quality & Verification Rules
- **Test and Retest Invariant**: Follow the [Mandatory Two-Cycle Verification Protocol](.agents/rules/test_and_retest.md) on every code change and deployment. Always execute two complete test cycles (Cycle 1: Implement -> Deploy -> Test; Cycle 2: Refine -> Redeploy -> Retest on warm state).
- **Behavioral Coding Guidelines**: Follow the [Behavioral Coding Guidelines](.agents/rules/coding_guidelines.md) ([`coding.md`](coding.md)) on every task. Bias toward caution over speed; for trivial tasks, use judgment.
- **Ollama CPU Resource Limits**: On GCE CPU VMs, keep `num_ctx: 4096` in `config.py` and avoid manual thread overrides (`num_thread: 16`), Flash Attention, or quantized KV caches to prevent thread oversubscription and CPU lockups.
- **Persistent Data Volume**: Keep all vector data, indexed files, and manifests located in `DATA_DIR` (`/mnt/disks/data` on VM). Never wipe persistent storage.

---

## Behavioral Coding Guidelines

### 1. Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First
**Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
**Touch only what you must. Clean up only your own mess.**
- When editing existing code:
  - Don't "improve" adjacent code, comments, or formatting.
  - Don't refactor things that aren't broken.
  - Match existing style, even if you'd do it differently.
  - If you notice unrelated dead code, mention it - don't delete it.
- When your changes create orphans:
  - Remove imports/variables/functions that YOUR changes made unused.
  - Don't remove pre-existing dead code unless asked.
- **The test**: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
**Define success criteria. Loop until verified.**
- Transform tasks into verifiable goals:
  - "Add validation" → "Write tests for invalid inputs, then make them pass"
  - "Fix the bug" → "Write a test that reproduces it, then make it pass"
  - "Refactor X" → "Ensure tests pass before and after"
- For multi-step tasks, state a brief plan:
  ```
  1. [Step] → verify: [check]
  2. [Step] → verify: [check]
  3. [Step] → verify: [check]
  ```
- Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

