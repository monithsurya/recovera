# Local language-model guidance

Recovera treats an LLM as a strategy selector. It does not use the model as a controller. The provider receives a bounded request containing the supported failure event, task phase, relevant observed state, scene revision, recovery history, and a trusted set of candidate strategies.

The response schema can identify one offered strategy, provide a short rationale, or abstain. It cannot provide coordinates, joint values, code, or a trajectory. Trusted application code checks the proposal against the request identity, current snapshot, candidate allowlist, and strategy prerequisites. Accepted strategies still pass through the existing kinematics, planner, trajectory validator, actuator execution, and state-based verification. Invalid, stale, uncertain, or infeasible proposals are not executed.

## Providers

- **Ollama:** Sends requests only to an HTTP loopback endpoint (`localhost`, `127.0.0.1`, or `::1`). Recovera does not download models. The selected model must already be available in Ollama.
- **Mock:** A deterministic, model-free provider for repeatable tests and demonstrations. Results identify this mode as `mock`; do not present them as real model decisions.
- **Deterministic fallback:** When enabled, provider failures or invalid proposals can fall back to the application's fixed policy. Use `--llm-fallback stop` to stop instead. Explicit model abstention, stale state, and failed planning checks stop without forcing a strategy.

Example:

```bash
ollama serve
ollama pull llama3.2:3b
recovera --check-llm --llm-provider ollama --llm-model llama3.2:3b
recovera --scenario drop-demo --llm --llm-provider ollama --llm-model llama3.2:3b
```

To run the same scenario with fixed mock inference:

```bash
recovera --scenario drop-demo --llm-provider mock
```

Logs and JSON results record provider, model, inference mode, latency, candidates, proposal disposition, planning result, and verified outcome. They do not print the full prompt or model rationale to the terminal. Provider timeout, trial-wide time budget, recovery-attempt cap, and candidate exclusions bound repeated work.

## Scope

The request contains privileged simulation observations, not a perception estimate. The model has no viewer, simulator, shell, robot, or tool access. A model response is untrusted input, and the proposal validator and planner—not the response—decide whether motion is permitted. Real-model success depends on the installed model and local service; mock-provider tests do not establish real-model performance.
