#!/bin/bash

source .venv/bin/activate
uv sync

uv run main.py dataset_name="openslr/librispeech_asr" subset_name="train.clean.100" output_path="./results" max_samples=5
