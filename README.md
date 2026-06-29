# Fidelo

Music generation project.

## Training

python .\train_latent_audio_tokenizer.py --epochs 20  
python .\train_latent_audio_prior.py --tokenizer-dir latent_audio_tokenizer_out --epochs 20      
python .\train_latent_audio_tokenizer.py --clip-seconds 3.33   

## Inference

python .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade.py --prompt "instrumental pop" --seed 88 --duration-seconds 30 --source-strength 0.85 --top-k 4 --top-p 0.95 --rank-choice-top 2 --window-energy-check-top 12 --min-window-rms 0.012 --min-window-peak 0.04 --theme-repeat-window 6 --theme-crossfade-ms 1000 --source-overlap 1024  --theme-repeat-bonus 3.5

python .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade.py --prompt "synth pop" --seed 1200 --duration-seconds 30 --source-strength 0.85 --top-k 4 --top-p 0.95 --rank-choice-top 2 --window-energy-check-top 32 --min-window-rms 0.16 --min-window-peak 0.55 --clip-energy-check-seconds 0.75 --min-clip-rms 0.14 --min-clip-peak 0.45 --min-clip-median-rms 0.3 --clip-retry-count 24 --theme-repeat-window 6 --theme-crossfade-ms 1000 --source-overlap 1024 --theme-repeat-bonus 3.5  

python .\generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_structured.py --prompt "classic rock" --seed 2 --duration-seconds 30 --source-strength 0.85 --top-k 4 --top-p 0.95 --rank-choice-top 2 --window-energy-check-top 12 --min-window-rms 0.012 --min-window-peak 0.04 --theme-repeat-window 6 --theme-crossfade-ms 1000 --source-overlap 1024 --theme-repeat-bonus 3.5 --intro-ratio 0.2 --outro-ratio 0.2 --intro-theme-top-n 1 --outro-theme-top-n 1 --intro-theme-seconds 2.8 --outro-theme-seconds 3.2 --intro-repeat-bonus 6.0 --outro-repeat-bonus 7.0 --song-intro-fade-ms 220 --song-outro-fade-ms 2200

## Model V2

After right-sizing the context length, model V2 quality improved.

|Evaluation|Model V1|Model V2|Improvement|
|-------------|----------------|----------------|------------------|
|Static average|1.2138|1.1482|5.36% less static|
|Static median|1.4004|1.2539 |10.43% less static|
|Silence|61 quiet rejects|44 quiet rejects|27.87% fewer rejects|

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
