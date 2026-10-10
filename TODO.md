# TODO

Open items from live voice testing on the Mac (2026-10-10).

## Late acknowledgement in live voice sessions

In a live `make run`, the question was routed at 30 s but "Just a minute, looking up rulings for…" was printed (and spoken) at 87 s. The user heard nothing and asked "What are you looking up?".

- Offline, with the same question, the real pipeline and real TTS but no mic, the acknowledgement fires instantly (2.3 s), so the cause is specific to the live mic or transcriber path.
- Suspects:
  - The Moonshine transcriber thread starving the worker. It runs keyterm-biased streaming STT on the same machine.
  - TTS synthesis contending with STT.
  - Output buffering in the test's log pipeline making the delay look worse than it was.
- Next step: instrument a live run with timestamps logged inside `JudgeListener.handle()` / `_acknowledge()` / `_speak()`, written to a file rather than piped through grep. Then check whether `tts.say()` itself is slow while the mic stream is active.

## Wrong answer: Blood Moon + Urza's Saga (retrieval)

"What happens when I play Blood Moon and my opponent has Urza's Saga?" The fallback now answers instead of going silent, but the answer is wrong: "Blood Moon becomes a nonbasic land…".

- The rules that decide the case don't reach the model. Those are CR 714.4 / 704.5s (only Sagas *with* chapter abilities are sacrificed), 714.3c (no lore counters without chapter abilities) and 305.7 (abilities granted by chapters are kept). The same gap fails `saga-blood-moon-current` in `tests/fixtures/hard_cases.json` for every model.
- The planned fix is the card/format cache. For the format's most-played cards (`corpus/metagame/<fmt>/cards.json`), precompute the CR sections each card depends on, offline on the Mac, and pin them whenever that card is named. Examples: Urza's Saga → 714, 704.5s, 305.7, 202.1b; Blood Moon → 305.7, 613.
- Measure with `make eval-hard`. The Urza's Saga topic is currently 32/48.

## Killing the judge mid-answer delays the next start

If the judge is stopped (Ctrl+C, or killed) while the LLM is generating, Ollama keeps finishing that orphaned request. The next startup's warm-load then queues behind it, which once cost ~3 minutes before the mic opened.

- The startup now prints "⏳ Loading the index and warming up …" so the wait is at least visible.
- Fix options:
  - Cancel the in-flight request on shutdown. Close the `ollama.Client` HTTP connection, or stream the response and stop reading.
  - Have startup check `GET /api/ps` and report that a previous generation is still running.

## Cloudflare R2 for data snapshots

Today snapshots publish to a Dropbox-synced folder (`~/Dropbox/mtg-judge-artifacts`), which a Pi without the Dropbox app can't easily download from. R2 would give every machine a private, scriptable download, with no egress fees for the ~530 MB index.

- Create a private R2 bucket (e.g. `mtg-judge-artifacts`) and two API tokens scoped to that bucket only: read/write for publishing from the Mac, and read-only for the Pi and other users.
- `scripts/publish_data.py`: add an S3-compatible backend (`boto3`, endpoint `https://<account_id>.r2.cloudflarestorage.com`) next to `LocalBackend`. Selected by an `MTG_JUDGE_ARTIFACTS=r2://<bucket>` style URL. Same layout: `latest.json` + `<version>/{manifest.json,index.tar.gz,corpus.tar.gz}`.
- `scripts/fetch_data.py`: read from R2 with the read-only token, or via presigned URLs so the Pi needs no credentials. Streaming download, sha256 verification and the swap-after-verify stay as they are.
- Credentials in environment variables or a git-ignored `.env`, never in the repo or config.
- Copy the existing Dropbox versions to R2 so `latest` matches. Then switch the defaults in `getting_started.sh`, the weekly update timer (fetch the latest snapshot rather than rebuilding on the Pi), and the README.
- Keep the bucket private: the snapshots contain Wizards' rules text verbatim.

## Nice to haves

### Voice fonts

Try different judge voices and pick a default for each profile. Today it's `piper_en_US-amy-medium` on the Mac and `-low` on the Pi (`tts_voice` in config).

- Moonshine offers 60 English voices across three engines:
  - 25 Piper voices (`piper_en_US-*`): light and fast, good for the Pi.
  - 20 Kokoro voices (`kokoro_af_*` / `kokoro_am_*`): more natural, heavier.
  - 15 ZipVoice voices (`zipvoice_*`): accents, plus voice cloning from a reference clip (`TextToSpeech.clone_from()`).
  - List them with `moonshine_voice.list_tts_voices("en_us")`.
- Measure synthesis time on the Pi for each candidate. The judge already waits on the LLM, so the voice shouldn't add noticeable delay.
- Ideas:
  - A `make voices` script that speaks a sample ruling in each voice.
  - A spoken "Judge, change your voice" command.
  - A cloned "head judge" voice. Only from audio we have rights to.

### Custom thinking sounds

Replace the bundled `success.wav` chime (Moonshine's `play_success()`) with our own sounds:
- an acknowledgement sound
- a "still thinking" keep-alive
- optionally, a soft ambient loop while the model reasons, stopped the instant the answer is ready
- an error tone

- Moonshine's earcons are fixed asset files. Custom sounds need their own playback, for example `sounddevice` on the same `output_device`. That playback must go through `JudgeListener._speak_lock` so it never overlaps speech, and it must keep the mic muted, or stay quiet enough that speech-to-text ignores it.
- Config could be `thinking_cue: custom` plus a `sounds/` folder (`ack.wav`, `thinking.wav`, `error.wav`), with the current cues as the fallback.
- Keep the files small and royalty-free (or self-made), and check they play on the Pi's audio output.
