#!/bin/bash

source .venv/bin/activate
uv sync

uv run main.py dataset_name="openslr/librispeech_asr" subset_name="clean[:10]" output_path="./results" max_samples=5
