# Engineering rules

- A single user message is not sufficient reason to add a production keyword or regex. First show that it represents a general failure class.
- Add new real phrasings to the evaluation dataset and regression tests before changing production behavior.
- Capture a baseline before substantial routing, prompt, model, or retrieval changes.
- Keep evaluation infrastructure and production behavior changes in separate tasks and commits.
- Make substantial architecture changes on a feature branch and verify them before merging into `main`.
- Never silently weaken citation validation, factual validation, grounding, or provider fallbacks.
- Ordinary `pytest` runs must not contact Gemini, Telegram, or other external services.
- Run live evaluation only by explicit command; report provider calls, latency, and errors.
- Document the cost and latency trade-off of any added LLM call.
- Never force push.
