# LLM benchmarks

## Current pick: `qwen3:8b` with thinking (2026-10-09)

The choice is optimized for accuracy, since latency matters less: a human judge also takes time. Every model got the same retrieval (k=4, ≤3,000 chars) and was graded on the full 38-case hard set. Thinking models get a 3,072-token output cap. Without it, one `qwen3:8b` answer once reasoned for 38 minutes.

**Screen (1 run each, 38 cases).** The cutoff for deletion was 10 points below the leader.

| Model | Correct | Fits Pi 8 GB | Kept |
|---|---|---|---|
| qwen3:8b, thinking | 28/38 | yes (5.2 GB) | ✓ |
| gpt-oss:20b, reasoning | 28/38 | no (13 GB) | ✓ (ceiling reference) |
| gemma2:9b | 27/38 | yes (5.4 GB) | ✓ |
| qwen3:4b, thinking | 26/38 | yes | ✓ |
| qwen3:14b, thinking | 26/38 | no | ✓ (ceiling reference) |
| qwen2.5:7b · phi4 (14B) | 24 · 24 | | deleted |
| qwen3:4b-instruct | 23 | | deleted |
| llama3.1:8b · qwen3:14b, no thinking | 22 · 22 | | deleted (14B kept for thinking) |
| gemma3:4b (previous default) · qwen3:8b, no thinking | 21 · 21 | | gemma3:4b deleted |
| gemma3:12b | 20 | | deleted |

Bigger isn't better here. The 12–20B models don't beat the 8–9B ones, which suggests the limit is the retrieved context, not model capacity. Earlier rounds had already ruled out qwen2.5:3b, llama3.2:3b and phi4-mini, which were deleted.

**Confirm (3 runs × 38 cases = 114).**

| Model | Correct | Median output tokens | Median time (Mac) | Strongest / weakest topics |
|---|---|---|---|---|
| **qwen3:8b, thinking** | **83/114 (73%)** | 540 | 14 s | Urza's Saga 32/48, sba 6/6 · commander 0/3, layers 5/12 |
| gemma2:9b | 79/114 (69%) | 78 | 5 s | replacement 18/18, layers 9/12 · Urza's Saga 24/48 |
| gpt-oss:20b, reasoning | 78/114 (68%) | 427 | 9 s | replacement 17/18 · combat 1/3 |

The three are within noise of each other: ±4 answers is about one standard error at n=114. qwen3:8b leads overall and on the Urza's Saga cases. **gemma2:9b is the fast alternative**: about 7× fewer output tokens and no reasoning, so it should be several times faster on a Pi.

**qwen3:8b profile checks (3 runs):**
- The actual Pi settings (4k context window, 3,000-token cap) score 82/114, the same as with 8k. The smaller window costs nothing.
- Twice the retrieved context (k=8, ≤6,000 chars) scores 78/114, no better, and layers drop from 6/12 to 3/12. Both profiles now use k=4 / ≤3,000 chars.

---

## Earlier rounds


Measured 2026-10-09 on a Mac (Apple Silicon, Ollama 0.40, default Q4 tags). The retrieval was identical for every model, using the Pi profile (`config/pi.yaml`: k=4, ≤3,000 chars of context). Rerun with:

```bash
bash scripts/benchmark_models.sh "gemma3:4b qwen2.5:7b"     # works on the Pi too
```

- **Hard set** (`tests/fixtures/hard_cases.json`) has 11 tricky interactions, each run 3 times (n=33). They cover Urza's Saga under the June 2025 Saga rules, Blood Moon, layers, replacement-effect order, ninjutsu timing and special actions. Answers are graded with a yes/no `verdict` and/or `answer_any` phrases. Every expected answer is grounded in the current CR or an official ruling.
- **Easy set** (`tests/fixtures/eval_cases.json`) has 20 basic questions. The score is the share of answers that cite an expected CR section.

| Model | Size | Hard set correct | Easy set cites | Median prompt / output tokens |
|---|---|---|---|---|
| **gemma3:4b** ← default | 3.3 GB | **21/33 (64%)** | 18/20 | 993 / 82 |
| qwen2.5:7b | 4.7 GB | 21/33 (64%) | 18/20 | 952 / 57 |
| qwen3:4b-instruct | 2.5 GB | 18/33 (55%) | 18/20 | 952 / 102 |
| llama3.1:8b | 4.9 GB | 18/33 (55%) | 18/20 | 907 / 50 |
| phi4-mini | 2.5 GB | 16/33 (48%) | 17/20 | 905 / 72 |
| llama3.2:3b | 2.0 GB | 11/33 (33%) | 17/20 | 917 / 47 |
| qwen2.5:3b (old Pi default) | 1.9 GB | 0/33 | 18/20 | 952 / 40 |

**Why gemma3:4b:** it ties qwen2.5:7b for accuracy at about 60% of the parameters. CPU inference on the Pi scales roughly with model size, so that matters. The Mac profile uses the same model, so desktop testing reflects what the Pi will say.

**Ruled out:**
- `qwen2.5:3b` fills the answer field with things like "high" or a bare rule number, even under schema-constrained output.
- `qwen3:4b` in thinking mode takes 24–55 s per answer on the Mac (minutes on a Pi), isn't more accurate on the Saga cases, and once aborted with "token repeat limit reached".

**Half the context** (k=3, ≤1,600 chars) cuts prompt tokens by about 35% (≈950 → ≈620). The effect on accuracy is mixed: gemma3:4b 21 → 17, qwen2.5:7b 21 → 21, phi4-mini 16 → 19, qwen3:4b-instruct 18 → 16. The default budget stays. If the Pi turns out too slow, lowering `max_context_chars` is the first lever to pull.

## What no model gets right

These three cases fail for every model:

- **Blood Moon + Urza's Saga (current rules).** Since June 2025, CR 714.4/704.5s only sacrifice Sagas *with* chapter abilities, so the Saga survives. Models either repeat the older 2021 ruling ("sacrificed after its last chapter") or answer vaguely. Several web sources, including a draftsim article, still give the old answer.
- **"Can they destroy the artifact chapter III will fetch?"** Chapter III doesn't target; the search happens on resolution. No targeting rule gets retrieved, and models reason about the trigger instead.
- **Crag Puca + Reckless Charge.** This is a reasoning failure, not a retrieval one. The context includes CR 613.4c/d and the exact Crag Puca ruling ("switch effects apply after all other effects"), yet every model computes the wrong P/T.

## Expanded Urza's Saga set (21 cases)

Ten more Urza's Saga cases were added, each verified against the CR and the oracle text or official rulings:
- chapter III and lands with no mana cost, and a {1} artifact
- Stifle and Tidebinder on chapter III
- Pithing Needle
- tapping for mana in response to chapter I
- the Construct counting itself
- Tron lands
- Blood Moon after chapter I
- Blood Moon leaving

The set now has 21 cases, 16 of them about Urza's Saga. `gemma3:4b` (Pi profile, ×3) scores **36/63 (57%)**. It gets 5 of the 10 new cases right 3/3 and the other 5 wrong 0/3, so its errors are consistent, not random:
- The Construct is "0/0" because "you control no artifacts". It doesn't count itself.
- Tidebinder "removes the Saga's abilities", even though the Saga isn't an artifact, creature or planeswalker.
- It "can tap for mana in response to chapter I", even though it has no mana ability yet.
- When Blood Moon leaves, it's "sacrificed because 2 ≥ 3".
- It omits {C} and/or {R} when asked about Blood Moon after chapter I.

## Broader hard set (38 cases, 11 topics)

Sixteen more cases cover classic hard rulings. Each was verified against CR text or an official Scryfall ruling, and the case records which one in `grounding`:
- **Layers:** Blood Moon + Urborg; Humility/Opalescence in both timestamp orders.
- **Replacement vs. cost/trigger:** Doubling Season with loyalty costs and entering loyalty; Panharmonicon with enters-with counters; Rest in Peace and dies triggers; Leyline of the Void + Living End.
- **Mana value:** cascade into a split card; Chalice vs. a split half on the stack; Thalia's tax.
- **Other:** split second and morph; indestructible vs. -X/-X; +1/+1 and -1/-1 counter annihilation; commander dies-then-command-zone; Lightning Greaves shroud; Teferi's Protection + Banishing Light.

`gemma3:4b` (Pi profile, ×3): **62/114 (54%)**, by topic:

| Topic | Correct |
|---|---|
| layers | 3/12 |
| urzas-saga | 21/48 |
| replacement | 8/18 |
| sba | 3/6 |
| mana-value | 6/9 |
| special-actions, commander, targeting, phasing, combat, zone-change | 3/3 or 6/6 each |

Layers and replacement-versus-trigger questions are the weak spots. The model also sometimes invents CR text: it credits "CR 702.16a" with Panharmonicon's wording, and claims "CR 712.21" makes exile count as going to the graveyard. When the cited number also appears in the retrieved context, citation verification lets these through.

While building this set, a data bug turned up and was fixed. Scryfall's oracle file represents some paper cards by an MTGO-only printing, Humility (Tempest Remastered) among them. The old "skip digital" filter had dropped about 1,000 real cards. Cards are now kept if they're Vintage- or Commander-legal.

## Pi latency

Not yet measured on the hardware. On CPU, latency is dominated by the ~950-token prompt, not the ~80-token answer. Run `bash scripts/benchmark_models.sh gemma3:4b` on the Pi and record the numbers here.
