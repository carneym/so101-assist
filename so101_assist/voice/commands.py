"""Parse transcripts into VoiceEvents.

Command words are a tiny, phonetically distinct, exact-match set:
    go, stop, grip, release, cancel, home, next mode
Everything else is treated as a target description and forwarded to
the detector's directed-query mode.

Rules:
- "stop" matches greedily anywhere in an utterance
- command matching is done on normalized text BEFORE any LLM/fuzzy
  step — safety words must not depend on a model call
- descriptions like "object two" resolve against current track_ids
"""
