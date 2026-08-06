"""Streaming speech-to-text via faster-whisper.

Runs a small model (base/small) locally with VAD segmentation.
Publishes raw transcripts; commands.py does the parsing.
Latency target: <700 ms from end of utterance to VoiceEvent,
because "stop" is a safety channel.
"""
