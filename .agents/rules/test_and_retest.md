# Rule: Mandatory Two-Cycle Verification and Retest Protocol

Whenever implementing changes, fixing bugs, or adding new features in this codebase:

## 1. Mandatory Two-Cycle Loop
Never declare a task complete after a single deploy or build. Always execute **two full test cycles**:

### Cycle 1 (Implement -> Local Validate -> Deploy -> Live Test)
1. **Pre-flight & Compilation**: Verify Python syntax (`python3 -m py_compile`) and run unit/component tests locally.
2. **Deploy to Target**: Copy modified code to the target environment (e.g. cloud VM / Docker container).
3. **Primary End-to-End Test**: Test all modified endpoints and affected UI tabs (Tabs 1–5) on the live instance.

### Cycle 2 (Refine Edge Cases -> Redeploy -> Retest on Warm State)
1. **Edge Case & Regression Check**: Test edge cases (e.g. empty inputs, file moves, multi-turn chat continuity, cache invalidation).
2. **Redeploy Updates**: Push any refinements or fixes to the target.
3. **Full Retest**: Retest all affected tabs and workflows on the deployed instance under warm cache/runtime conditions.

## 2. Verification Standards
* **Never Assume Success from Exit Codes**: A successful `docker restart` or `git checkout` does not prove functionality. Execute real queries, check inference output tokens, and inspect HTTP/API response bodies.
* **Test Across All Dependent Interfaces**: In multi-tab architectures (like Streamlit), changes to one component (e.g. ingestion) affect multiple tabs (search, citations, chat). Test all connected tabs, not just the one modified.
* **Document Test Evidence**: Always record duration, token throughput, and sample outputs in the walkthrough.
