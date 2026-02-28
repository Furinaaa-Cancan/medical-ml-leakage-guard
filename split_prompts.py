#!/usr/bin/env python3
"""Split QUEUE_PROMPTS.md into individual prompt files for easy queue loading."""
import re
from pathlib import Path

src = Path(__file__).parent / "QUEUE_PROMPTS.md"
out = Path(__file__).parent / "queue"
out.mkdir(exist_ok=True)

text = src.read_text(encoding="utf-8")

# Split on ### N patterns (### 1, ### 2, ..., ### 100)
parts = re.split(r"\n### (\d+)\n", text)
# parts = [preamble, "1", content1, "2", content2, ...]

count = 0
for i in range(1, len(parts) - 1, 2):
    num = int(parts[i])
    body = parts[i + 1].strip()
    # Remove trailing --- separator
    body = re.sub(r"\n---\s*$", "", body).strip()

    fname = out / f"{num:03d}.txt"
    fname.write_text(body + "\n", encoding="utf-8")
    count += 1

print(f"Done: {count} prompts → {out}/")
print(f"Files: {out}/001.txt .. {out}/{count:03d}.txt")
