#!/bin/bash

python eval.py \
--data_path ./data/AI/ai_lyrics.json \
--output_path ./results/results.json \
--model_path /data/project/model_weights/lyrics_gen/Qwen3-0.6B \
--batch_size 8
