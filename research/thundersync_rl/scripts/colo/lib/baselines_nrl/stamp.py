"""Prefix every stdin line with a wall-clock epoch timestamp ("[t=1730270000.123] ...")."""
import sys
import time

for line in iter(sys.stdin.buffer.readline, b""):
    sys.stdout.buffer.write(b"[t=%.3f] " % time.time() + line)
    sys.stdout.buffer.flush()
