# Mistral post-normalization safety loop plan

## Goal

After a candidate `sent_prompt` is built, review its remaining jailbreak or unsafe risk with Mistral instead of keyword/rule-based logic. If Mistral still sees risk, re-normalize the prompt and review again. Allow at most 3 refinement loops. If the prompt is still risky after those loops, send exactly `{BLOCKED}`.

## Design

1. Add a Mistral-only review prompt in `mistral_gate.py`.
   - It returns only `0` or `1`.
   - `0` means the prompt is safe enough to send.
   - `1` means the prompt still contains jailbreak, prompt injection, unsafe payload, hidden instruction, or unsafe task risk.
   - Parsing failures do not fall back to keyword heuristics. They are treated conservatively as risky.

2. Add staged re-normalization prompts in `mistral_gate.py`.
   - Stage 1 removes obvious jailbreak wrappers and unsafe instructions.
   - Stage 2 applies stricter filtering: remove roleplay, examples, quoted instructions, emotional pressure, and anything that tries to alter model behavior.
   - Stage 3 keeps only a short benign user task. If no clearly benign task remains, output exactly `{BLOCKED}`.

3. Add `MistralGate.normalize_until_safe(...)`.
   - Inputs: current candidate prompt, original prompt, optional review prompt, maximum loops.
   - Flow:
     1. Review the candidate with Mistral.
     2. If safe, return the candidate.
     3. If risky and fewer than 3 loops have been used, re-normalize with the next stronger stage.
     4. If still risky after 3 loops, return `{BLOCKED}`.
   - Return metadata containing review outputs, normalization outputs, loop count, and whether blocking happened.

4. Preserve `{BLOCKED}` in platform preparation.
   - `ensure_alignment_arena_prompt()` must return `{BLOCKED}` unchanged instead of appending `{{ unsafe_request }}`.

5. Update call sites.
   - `mistral_test.py`: after the initial classify/normalize step builds `sent_prompt_base`, run `normalize_until_safe()` before posting.
   - `dreadnode_final.py`: same change inside the after-input Mistral gate pipeline.
   - Store the new loop metadata in JSON outputs so results explain why a prompt was sent, refined, or blocked.

## Verification

Run Python syntax compilation for the touched files:

```powershell
python -m py_compile .\---\mistral_gate.py .\---\mistral_test.py .\---\dreadnode_final.py
```
