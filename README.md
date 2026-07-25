# Fidelo

Music generation project.

## Training

python .\train_latent_audio_tokenizer.py --epochs 20  
python .\train_latent_audio_prior.py --tokenizer-dir latent_audio_tokenizer_out --epochs 20      
python .\train_latent_audio_tokenizer.py --clip-seconds 3.33   
python .\train_latent_audio_tokenizer.py --finetune-from latent_audio_tokenizer_out --epochs 5
python .\train_latent_audio_tokenizer.py --finetune-from  latent_audio_tokenizer_out  --lr 5e-5 --weight-decay 5e-6 --grad-accum-steps 4 --commitment-cost 0.2 --clip-seconds 5 --num-quantizers 2 --epochs 5


python .\prepare_beginning_transition_manifest.py `
  --metadata-csv .\dataset\metadata.csv `
  --audio-dir .\dataset\audio `
  --output .\dataset\beginning_transition_manifest.csv `
  --intro-seconds 10 `
  --body-seconds 9

python.exe `
  .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_intro_body_continued.py `
  --tokenizer-dir .\3.33smodel\9smodel `
  --prior-dir .\3.33smodel\9smodel\prior_transition_finetuned `
  --prompt "your prompt" `
  --duration-seconds 45 `
  --continuation-prefix-seconds 2.0 `
  --output .\inference_output\continued_song.wav

The continued generator carries the exact GRU hidden state across internal windows and accepted output clips. After source retrieval and token fusion, it advances the state with the actual accepted tokens, so recurrent context matches the audio codes that are retained. Every retry branches from the last accepted state; a rejected retry does not alter later generation. `--continuation-prefix-seconds` remains useful for source-window continuity, while the GRU itself uses persistent state by default. Add `--disable-persistent-gru-state` to restore prefix-replay behavior for comparison.

For a thematic transition that guides token generation instead of blending two completed clips, use the latent-guided token variant:

```powershell
python.exe `
  .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_intro_body_latent_guided_tokens.py `
  --tokenizer-dir .\3.33smodel\9smodel `
  --prior-dir .\3.33smodel\9smodel\prior_transition_finetuned `
  --prompt "metal" `
  --duration-seconds 45 `
  --latent-transition-seconds 4.0 `
  --latent-motif-match-seconds 0.75 `
  --latent-transition-min-ac-rms 0.008 `
  --latent-overlap-neighborhood-seconds 0.08 `
  --latent-overlap-neighborhood-candidates 9 `
  --latent-overlap-continuity-weight 0.35 `
  --latent-overlap-offset-weight 0.05 `
  --latent-overlap-progression-weight 0.5 `
  --clip-crossfade-ms 900 `
  --section-crossfade-ms 1800 `
  --continuation-prefix-seconds 2.0 `
  --output .\inference_output\latent_guided_song.wav
```

This variant retrieves regular-source motifs by similarity to the accepted intro tail and rejects quiet motif windows before constructing the transition. It creates a smoothly evolving latent target across the aligned intro and regular overlap. A global path optimizer selects intact residual-VQ pairs from nearby observed positions, scoring target distance, adjacent latent jumps, distance from aligned source timing, and skips or repeats within each source. The path starts in the intro, switches to the regular motif exactly once, and cannot return to the intro. It never averages token IDs, combines quantizer streams from different candidates, or inserts arbitrary prior tokens inside the overlap.

`--latent-overlap-neighborhood-seconds` bounds how far selection may move from each aligned source position, and `--latent-overlap-neighborhood-candidates` controls how many positions per source are evaluated. Increase `--latent-overlap-continuity-weight` to favor a cleaner splice boundary, `--latent-overlap-offset-weight` to stay near aligned timing, and `--latent-overlap-progression-weight` to discourage skipped or repeated source tokens. Set the neighborhood to `0` for a strict splice between two contiguous observed runs. These options are separate from the source-window retrieval option `--continuity-weight`. Selection is deterministic for fixed inputs.

Every selected pair is committed to persistent GRU state before ordinary body generation resumes. Accepted clips are decoded separately and joined with equal-power waveform overlaps. The saved output preserves the internal themed segment fades controlled by `--theme-crossfade-ms`, rather than applying them only during energy checking. `--clip-crossfade-ms` controls every ordinary clip/sequence boundary. `--section-crossfade-ms` controls the intro-to-regular overlap and is centered on the selected token path's first regular-motif pair. The duplicated intro-like prefix of the body transition is skipped before mixing, so the fade spans the actual musical handoff instead of ending before regular tokens arrive. Duration-based generation automatically requests enough clips to compensate for these overlaps before trimming to the requested output length. Mean-centered RMS validation rejects quiet static or DC-like transition output; tune that check with `--latent-transition-min-ac-rms`.
  
### Song-beginning conditioning

The latent prior supports a dedicated input-only BOS token for clips taken from the actual start of a song. Add a `song_beginning` column to the prior-training metadata:

```csv
file,text,song_beginning
intro_001.mp3,"genre rock; energetic guitars",1
regular_001.mp3,"genre rock; energetic guitars",0
```

Accepted true values are `1`, `true`, `yes`, `y`, `beginning`, and `start`. Missing or empty values are treated as regular clips. Beginning-labelled files are always cropped from sample zero. The audio tokenizer does not need retraining; fine-tune only the latent prior and keep the saved text vocabulary beside the checkpoint:

```powershell
python .\train_latent_audio_prior.py `
  --tokenizer-dir latent_audio_tokenizer_out `
  --metadata-csv dataset/metadata_with_beginnings.csv `
  --audio-dir dataset/audio `
  --finetune-from latent_audio_prior_out `
  --out-dir latent_audio_prior_beginning_out `
  --lr 5e-5 `
  --epochs 10
```

Generation scripts use beginning BOS for the first output clip by default. Use `--beginning-bos-clips N` to apply it to the first N clips, or `--beginning-bos-clips 0` to disable it:

```powershell
python .\generate_latent_audio_cuda.py `
  --prompt "classic rock" `
  --duration-seconds 30 `
  --beginning-bos-clips 3
```

This changes how each selected clip starts. Later windows inside a clip continue to use their latent prefix rather than injecting beginning BOS again.

In the structured generator, retrieval sources marked `song_beginning=1` are restricted to the intro section. Body and outro sections select only regular sources. `--beginning-bos-clips` controls the prior's BOS seed count; it does not allow beginning-labelled retrieval audio in later song sections.

The structured generator can use quieter energy gates for clips that receive song-beginning BOS. Omitted beginning overrides inherit the regular value:

```powershell
--window-energy-check-top 12 `
--min-window-rms 0.004 `
--min-window-peak 0.018 `
--clip-energy-check-seconds 2.0 `
--min-clip-rms 0.006 `
--min-clip-peak 0.028 `
--beginning-window-energy-check-top 12 `
--beginning-min-window-rms 0.001 `
--beginning-min-window-peak 0.006 `
--beginning-clip-energy-check-seconds 2.0 `
--beginning-min-clip-rms 0.002 `
--beginning-min-clip-peak 0.010
```

These overrides follow the actual BOS assignment, including a body clip covered by `--beginning-bos-clips N`. They affect both retrieved source-window checks and final generated-clip acceptance.

### Single-stream tokenizer fine-tuning
- A single-stream tokenizer automatically gains a second residual quantizer. This upgrade intentionally applies only to single-stream tokenizers; choose higher stream counts explicitly with `--num-quantizers`.
- The first two epochs freeze the encoder and existing quantizer while training the new stream and decoder.
- The default learning rate is `5e-5`; override it with `--lr`.
- Pass `--num-quantizers 1` to retain a single stream, or `--residual-finetune-warmup-epochs 0` to retain the prior no-warmup behavior.

## Inference

python .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade.py --prompt "instrumental pop" --seed 88 --duration-seconds 30 --source-strength 0.85 --top-k 4 --top-p 0.95 --rank-choice-top 2 --window-energy-check-top 12 --min-window-rms 0.012 --min-window-peak 0.04 --theme-repeat-window 6 --theme-crossfade-ms 1000 --source-overlap 1024  --theme-repeat-bonus 3.5

python .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade.py --prompt "synth pop" --seed 1200 --duration-seconds 30 --source-strength 0.85 --top-k 4 --top-p 0.95 --rank-choice-top 2 --window-energy-check-top 32 --min-window-rms 0.16 --min-window-peak 0.55 --clip-energy-check-seconds 0.75 --min-clip-rms 0.14 --min-clip-peak 0.45 --min-clip-median-rms 0.3 --clip-retry-count 24 --theme-repeat-window 6 --theme-crossfade-ms 1000 --source-overlap 1024 --theme-repeat-bonus 3.5  

python .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_structured.py --prompt "classic rock" --seed 2 --duration-seconds 30 --source-strength 0.85 --top-k 4 --top-p 0.95 --rank-choice-top 2 --window-energy-check-top 12 --min-window-rms 0.012 --min-window-peak 0.04 --theme-repeat-window 6 --theme-crossfade-ms 1000 --source-overlap 1024 --theme-repeat-bonus 3.5 --intro-ratio 0.2 --outro-ratio 0.2 --intro-theme-top-n 1 --outro-theme-top-n 1 --intro-theme-seconds 2.8 --outro-theme-seconds 3.2 --intro-repeat-bonus 6.0 --outro-repeat-bonus 7.0 --song-intro-fade-ms 220 --song-outro-fade-ms 2200

python .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_structured.py `
  --prompt "electropop" `
  --seed 54367 `
  --duration-seconds 30 `
  --source-strength 0.85 `
  --top-k 4 `
  --top-p 0.95 `
  --rank-choice-top 2 `
  --window-energy-check-top 12 `
  --min-window-rms 0.004 `
  --min-window-peak 0.018 `
  --clip-energy-check-seconds 2.0 `
  --min-clip-rms 0.006 `
  --min-clip-peak 0.028 `
  --clip-retry-count 8 `
  --theme-repeat-window 6 `
  --theme-crossfade-ms 1000 `
  --source-overlap 1024 `
  --theme-repeat-bonus 3.5 `
  --intro-ratio 0.2 `
  --outro-ratio 0.2 `
  --intro-theme-top-n 1 `
  --outro-theme-top-n 1 `
  --intro-theme-seconds 2.8 `
  --outro-theme-seconds 3.2 `
  --intro-repeat-bonus 6.0 `
  --outro-repeat-bonus 7.0 `
  --song-intro-fade-ms 220 `
  --song-outro-fade-ms 2200

python .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_structured.py `
  --prompt "electropop" `
  --seed 123 `
  --duration-seconds 30 `
  --source-strength 0.58 `
  --temperature 1.0 `
  --top-k 8 `
  --top-p 0.92 `
  --repetition-penalty 1.08 `
  --rank-choice-prob 0.4 `
  --rank-choice-top 4 `
  --rank-choice-temperature 0.8 `
  --creative-span-count 6 `
  --creative-span-min 12 `
  --creative-span-max 36 `
  --creative-token-mix 0.24 `
  --window-energy-check-top 12 `
  --min-window-rms 0.004 `
  --min-window-peak 0.018 `
  --clip-energy-check-seconds 2.0 `
  --min-clip-rms 0.006 `
  --min-clip-peak 0.028 `
  --clip-retry-count 8 `
  --theme-top-n 8 `
  --theme-temperature 1.0 `
  --theme-repeat-window 4 `
  --theme-crossfade-ms 1000 `
  --source-overlap 128 `
  --theme-repeat-bonus 1.5 `
  --intro-ratio 0.2 `
  --outro-ratio 0.2 `
  --intro-theme-top-n 2 `
  --outro-theme-top-n 2 `
  --intro-theme-seconds 2.8 `
  --outro-theme-seconds 3.2 `
  --intro-repeat-bonus 3.0 `
  --outro-repeat-bonus 4.0 `
  --intro-source-strength 0.68 `
  --outro-source-strength 0.72 `
  --intro-creative-token-mix 0.18 `
  --outro-creative-token-mix 0.14 `
  --intro-rank-choice-prob 0.24 `
  --outro-rank-choice-prob 0.18 `
  --song-intro-fade-ms 220 `
  --song-outro-fade-ms 2200
  
  python .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_intro_body_latent_guided_tokens.py `
  --prompt "synthwave" `
  --seed 178 `
  --duration-seconds 45 `
  --repetition-penalty 1.03 `
  --repetition-window 256 `
  --rank-choice-prob 0.08 `
  --rank-choice-top 2 `
  --rank-choice-temperature 0.30 `
  --source-candidates 16 `
  --max-source-seconds 30 `
  --proposal-weight 0.70 `
  --continuity-weight 5.0 `
  --match-weight 0.65 `
  --scan-step-divisor 8 `
  --creative-span-count 2 `
  --creative-span-min 8 `
  --creative-span-max 20 `
  --creative-token-mix 0.04 `
  --theme-top-n 2 `
  --theme-temperature 0.20 `
  --theme-repeat-window 8 `
  --theme-repeat-bonus 7.0 `
  --theme-repeat-decay 0.85 `
  --theme-crossfade-ms 600 `
  --intro-ratio 0.20 `
  --intro-theme-top-n 1 `
  --intro-theme-temperature 0.15 `
  --intro-repeat-bonus 8.0 `
  --intro-creative-token-mix 0.02 `
  --intro-rank-choice-prob 0.03 `
  --song-intro-fade-ms 450 `
  --clip-retry-count 8 `
  --intro-body-prior-seconds 0 `
  --intro-body-average-seconds 0.4 `
  --intro-body-overlap-seconds 1.5 `
  --intro-body-source-strength 1.0 `
  --fade-ms 1200 `
  --latent-transition-seconds 4.0 `
  --latent-motif-match-seconds 0.75 `
  --latent-guidance-strength 4.0 `
  --latent-guidance-candidate-top-k 4 `
  --latent-guidance-temperature 0 `
  --continuation-prefix-seconds 2.0 `
  --source-strength 0.92 `
  --source-window 512 `
  --source-overlap 512 `
  --temperature 0.60 `
  --top-k 4 `
  --top-p 0.85 `
  --intro-theme-seconds 4.0 `
  --theme-seconds 4.0 `
  --window-energy-check-top 12 `
  --min-window-rms 0.0035 `
  --min-window-peak 0.015 `
  --clip-energy-check-seconds 2.0 `
  --min-clip-rms 0.006 `
  --min-clip-peak 0.028 `
    --window-energy-check-top 64 `
  --latent-overlap-neighborhood-seconds 0 `
  --latent-overlap-neighborhood-candidates 1 `
  --latent-overlap-continuity-weight 2.0 `
  --latent-overlap-progression-weight 5.0 `
  --latent-transition-min-ac-rms 0.008 `   
  --latent-transition-seconds 6.0 `
  --section-crossfade-ms 3500 `
  --theme-crossfade-ms 900 `
  --clip-crossfade-ms 1500
  
## Model V2

After right-sizing the context length, model V2 quality improved.

|Evaluation|Model V1|Model V2|Improvement|
|-------------|----------------|----------------|------------------|
|Static average|1.2138|1.1482|5.36% less static|
|Static median|1.4004|1.2539 |10.43% less static|
|Silence|61 quiet rejects|44 quiet rejects|27.87% fewer rejects|

Fine tune with 5 second context.

## Model V3

Fine tune with 4 gradient accumulation steps, 5 second context, and decreased 5e-5 learning rate.

Model V3 reconstructs audio 1.43% more faithfully than Model V2.

## Model V4

Fine tune with lower weight decay, higher commitment cost, 4 gradient accumulation steps, and 5e-5 learning rate.

Model V4 reconstructs audio 1.69% more faithfully than Model V3.

## Model V5

Model V5 reconstructs audio 0.52% more faithfully than Model V4.

## Model V6

Model V6 reconstructs audio 12.07% more faithfully than Model V1.

## Dataset Preparation

Started with a 10,000 sample subset from a larger Free Music Archive audio clip and description dataset.

Used local Gemma 4, Ollama and OpenCode to create Python script that uses Gemma 4 as a judge of dataset audio descriptions to filter out rows where a person is speaking, talking, narrating, giving dialogue, being interviewed, making a speech, conversation, podcast, announcement, commentary, or other spoken words.

# Latent Music Generation Architecture

This project uses a **two-stage music generation system**.

Instead of generating raw audio samples directly, it first turns audio into **latent tokens**. You can think of latent tokens as a compressed musical shorthand.

That makes the problem easier:
- the **tokenizer** learns how to compress and rebuild audio
- the **prior** learns how to predict good token sequences from text prompts
- the **generator** turns predicted tokens back into waveform audio

---

## High-level idea

The system works like this:

1. **Take real audio from the dataset**
2. **Compress it into latent codes**
3. **Train a model to reconstruct the original audio from those codes**
4. **Train a second model to predict those codes from text**
5. **At inference time, predict new codes from a prompt**
6. **Decode the predicted codes back into audio**

So the model is not directly writing every audio sample.
It is writing a compact sequence of learned audio symbols first.

---

## Main parts

### 1. Audio tokenizer

File: [latent_audio_token_pipeline.py](latent_audio_token_pipeline.py)

The tokenizer is an **autoencoder with vector quantization**.

It has 3 main jobs:

- **Encoder**: reads waveform audio and compresses it into a smaller hidden representation
- **Quantizer**: snaps that representation to the nearest learned codebook entry
- **Decoder**: turns those discrete codes back into waveform audio

In simple terms:
- encoder = compress
- quantizer = turn compression into discrete tokens
- decoder = rebuild sound

Why this matters:
- if reconstruction sounds good, the token representation is useful
- if reconstruction sounds bad, generation will also sound bad

### 2. Text-conditioned prior

Files:
- [latent_audio_token_pipeline.py](latent_audio_token_pipeline.py)
- [train_latent_audio_prior.py](train_latent_audio_prior.py)

The prior is the model that learns:

> “Given this text prompt, what latent token should come next?”

It does **not** generate waveform audio directly.
It only generates token IDs.

It reads:
- text prompt tokens
- previous latent tokens

And predicts:
- the next latent token

In this project, the prior is a recurrent sequence model that conditions on a pooled text representation.

### 3. Inference / generation

File: [generate_latent_audio_cuda.py](generate_latent_audio_cuda.py)

At generation time:

1. the prompt text is tokenized
2. the prior predicts a sequence of latent codes
3. the tokenizer decoder converts those codes into waveform audio
4. output clips are stitched together into a final WAV file

There is also **retrieval guidance** during inference.
That means the generator can look at prompt-matched dataset examples and bias generation toward latent patterns that resemble real training audio.

This helps reduce the “just a few simple notes” problem.

---

## Training pipeline

### Stage 1: Train the tokenizer

File: [train_latent_audio_tokenizer.py](train_latent_audio_tokenizer.py)

Goal:
- learn a latent representation that can reconstruct the input audio well

The tokenizer is trained with losses that encourage:
- sample-level waveform similarity
- spectral similarity
- stable vector quantization

If this stage works well:
- reconstructed clips should sound close to the original dataset audio

If this stage works poorly:
- the prior has no good token language to learn from

### Stage 2: Train the prior

File: [train_latent_audio_prior.py](train_latent_audio_prior.py)

Goal:
- learn the sequence structure of latent codes
- connect text descriptions to likely token patterns

This stage uses the trained tokenizer to convert real dataset audio into codes.
Then it trains the prior to predict those codes autoregressively.

---

## Why use latent tokens instead of raw waveform generation?

Generating raw waveform samples directly is very hard.
A few problems are:
- sequences are extremely long
- local sample prediction often produces weak or noisy sound
- the model spends too much effort on tiny waveform details

Latent tokens help because they:
- shorten the sequence length
- force the model to learn higher-level audio structure
- make text-conditioned generation more practical

So instead of predicting millions of sample values, the model predicts a much smaller set of meaningful learned codes.

---

## What “good” behavior looks like

### Good tokenizer behavior
- reconstructed audio sounds very similar to the original
- transients, tone, and texture are preserved
- tokenized audio still feels like real music

### Good prior behavior
- generated code sequences sound structured
- output is not just repeated tones or simple hums
- prompt words influence style and texture
- decoded audio resembles real dataset clips in complexity

---

## Current weak point in systems like this

Usually the hardest part is the **tokenizer**.

Why:
- if the tokenizer throws away too much information, no prior can fix it
- if the codebook is weak, decoded audio becomes blurry or overly simple
- if compression is too strong, music loses texture and detail

So in practice:
- reconstruction quality is the first thing to improve
- only after that does better prior training really matter

---

## Simple mental model

A simple way to think about the whole system:

- **Tokenizer** = learns a compact alphabet of sound pieces
  - latent_audio_token_pipeline.py and train_latent_audio_tokenizer.py
- **Prior** = learns how to arrange those pieces from text
  - latent_audio_token_pipeline.py and train_latent_audio_prior.py
- **Decoder** = turns those arranged pieces back into audio
  - latent_audio_token_pipeline.py and used at inference in generate_latent_audio_cuda.py

Or even shorter:

- compress music
- learn the compressed language
- generate new compressed sequences
- decode back to sound

---

## Important files

- [latent_audio_token_pipeline.py](latent_audio_token_pipeline.py) — shared model components
- [train_latent_audio_tokenizer.py](train_latent_audio_tokenizer.py) — tokenizer training
- [train_latent_audio_prior.py](train_latent_audio_prior.py) — prior training
- [generate_latent_audio_cuda.py](generate_latent_audio_cuda.py) — inference
- [test_latent_tokenizer_reconstruction.py](test_latent_tokenizer_reconstruction.py) — reconstruction quality test

---

## Practical summary

If you want better final music quality, the usual order is:

1. improve tokenizer reconstruction
2. verify reconstructed clips sound close to real audio
3. train the prior on those better tokens
4. tune inference so predicted tokens decode clearly

That is the core architecture of the latent music generation pipeline in this project.
