# Conversation routing annotation guide

The expected action describes what the assistant should do next. It does not
prescribe exact response wording.

- `SEARCH_KNOWLEDGE`: A factual domain question about UniME, ERSU, or student
  life in Messina, including a follow-up whose subject is clear from history.
- `DIRECT_REPLY`: A message that needs a conversational response without a
  knowledge search, such as a greeting, thanks, farewell, standalone emotion,
  or standalone profanity.
- `ASK_CLARIFICATION`: A potentially domain-related question that is too vague
  to resolve reliably from its message and history.
- `EXPLAIN_PREVIOUS`: A request to explain the previous assistant answer or
  system outcome.
- `OUT_OF_SCOPE`: A clearly unrelated request, or a follow-up that clearly
  refers to an unrelated topic in history.

Annotate from the message and supplied history. Do not tune labels to the
current router or interpreter output.
