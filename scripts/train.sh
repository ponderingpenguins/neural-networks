#!/bin/bash

source .venv/bin/activate
uv sync

# Stage 1: encode the audio with Whisper once and cache the speech embeddings to disk
uv run main.py stage=bake dataset_name="openslr/librispeech_asr" dataset_split="train.clean.100" output_path="./results" max_samples=1

# Stage 2: train the projector on the cached embeddings
uv run main.py stage=train dataset_name="openslr/librispeech_asr" dataset_split="train.clean.100" output_path="./results" max_samples=1 batch_size=1
