# Libraries Used for Training and Inference

This project is built around a fairly small library stack. Most of the heavy lifting is done by PyTorch, while audio loading, saving, and dataset/config handling are kept simple.

## Core Training Libraries

### `torch`

This is the main machine learning library in the project.

Why it is used:
- Defines the tokenizer and prior models.
- Runs tensor math on GPU for training and inference.
- Provides autograd for backpropagation.
- Provides optimizers, losses, checkpoint loading, and checkpoint saving.
- Provides `torch.stft`, which is used in tokenizer training for spectral reconstruction loss.

Where it shows up:
- [train_latent_audio_tokenizer.py](d:/Projects/music4/train_latent_audio_tokenizer.py)
- [train_latent_audio_prior.py](d:/Projects/music4/train_latent_audio_prior.py)
- [latent_audio_token_pipeline.py](d:/Projects/music4/latent_audio_token_pipeline.py)
- [generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_structured.py](d:/Projects/music4/generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_structured.py)

Why it fits this repo:
- The whole system is a learned latent-audio pipeline, so tensor operations, neural network modules, and GPU execution are the central requirement.

### `torch.nn` and `torch.nn.functional`

These are PyTorch submodules used to build the actual network layers and losses.

Why they are used:
- `torch.nn` provides modules like convolutions, recurrent layers, embeddings, normalization layers, and linear layers.
- `torch.nn.functional` is used for functional building blocks and loss calculations.

Why they fit this repo:
- The tokenizer is a neural audio autoencoder with vector quantization.
- The prior is a text-conditioned sequence model over latent codes.

### `torch.utils.data`

This is PyTorch’s dataset and batching API.

Why it is used:
- `Dataset` is used to represent paired audio/text training samples.
- `DataLoader` handles minibatching and iteration.
- `random_split` is used for train/validation splits.
- `default_collate` is used inside the custom safe collate path.

Why it fits this repo:
- Training needs a stable way to stream many fixed-length waveform/text examples without manually batching tensors.

### `tqdm`

This library provides progress bars in the terminal.

Why it is used:
- Shows training progress during tokenizer and prior training loops.
- Makes long-running GPU training easier to monitor.

Why it fits this repo:
- These training jobs are iterative and can run for a long time, so progress feedback is useful without changing core model code.

## Core Inference Libraries

### `torch`

PyTorch is also the central inference library.

Why it is used during inference:
- Loads the trained tokenizer and prior checkpoints.
- Generates latent token sequences from prompts.
- Decodes latent codes back into waveforms.
- Computes energy and loudness gating statistics on generated audio.

Why it fits this repo:
- Inference is not a separate stack here. It reuses the same model definitions and tensor operations as training.

### `soundfile`

This library is used for audio file I/O when available.

Why it is used:
- Reads waveform audio from dataset files.
- Writes generated audio back to disk as WAV files.

Where it shows up:
- [latent_audio_token_pipeline.py](d:/Projects/music4/latent_audio_token_pipeline.py)

Why it fits this repo:
- It is a lightweight and practical option for reading and writing PCM audio without pulling in a much larger dependency stack.

### `torchaudio`

This is used as a fallback for audio input/output if `soundfile` is not available or fails.

Why it is used:
- Loads audio files into tensors.
- Saves tensors as audio files.
- Keeps audio I/O in the PyTorch ecosystem when needed.

Why it fits this repo:
- The project already uses PyTorch tensors everywhere, so `torchaudio` is a natural fallback path.

## Shared Data and Configuration Libraries

### `csv`

Used to read the dataset metadata.

Why it is used:
- Reads [dataset/metadata.csv](d:/Projects/music4/dataset/metadata.csv) to map audio files to text descriptions.

Why it fits this repo:
- The dataset format is simple and does not need a heavier table-processing dependency.

### `json`

Used for configuration and tokenizer serialization.

Why it is used:
- Saves model configs.
- Saves and loads the simple text tokenizer vocabulary.

Why it fits this repo:
- Model bundles here are simple file-based artifacts, so JSON is enough for human-readable config storage.

### `os`

Used for path handling and file checks.

Why it is used:
- Builds paths for checkpoints, config files, dataset audio, and outputs.
- Checks whether candidate files exist.

Why it fits this repo:
- The project is script-oriented and file-based, so standard library path/file utilities are sufficient.

### `argparse`

Used by the CLI scripts.

Why it is used:
- Defines training and inference flags.
- Lets you tune model, dataset, retrieval, gating, and structure behavior from the command line.

Why it fits this repo:
- This codebase is organized around standalone Python scripts rather than a packaged application UI.

## Model-Specific Internal Code

### `latent_audio_token_pipeline.py`

This is not an external library, but it is the shared internal module everything else depends on.

Why it exists:
- Centralizes model definitions.
- Centralizes dataset loading.
- Centralizes audio loading/saving helpers.
- Centralizes checkpoint bundle loading and saving.

Why it matters:
- Training and inference stay aligned because they use the same config format, tokenizer architecture, prior architecture, and I/O helpers.

## Libraries Notably Not Central Here

### `numpy`

There is no top-level direct dependency on NumPy in the main training and inference scripts.

Why that matters:
- Most computation stays in PyTorch tensors instead of bouncing between tensor and array libraries.
- Audio I/O helpers can still interact with NumPy indirectly through `soundfile`, but the project’s main compute path is PyTorch-native.

### `pandas`

Not used for the main training/inference path.

Why that matters:
- Metadata handling is simple enough that the standard `csv` module is sufficient.

## Practical Summary

If you reduce the stack to the essentials, it looks like this:

- `torch`: models, training, inference, tensor math, checkpointing.
- `torch.utils.data`: datasets, dataloaders, train/validation splits.
- `tqdm`: training progress display.
- `soundfile` and `torchaudio`: audio input/output.
- `csv` and `json`: metadata and config/tokenizer serialization.
- `argparse`: command-line control over all scripts.

That is a sensible stack for this repo because the project is fundamentally a PyTorch latent-audio research workflow with lightweight script-based orchestration around it.