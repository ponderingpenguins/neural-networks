# neural-networks

## Quick-lookup

paper: https://arxiv.org/pdf/2402.08846
code: https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/asr_librispeech


## Micro abstract



Ma et al. (2024) systematically explored how LLM-based ASR systems can be constructed by only training a linear projector to align the speech encoder and the LLM. However, the paper did not analyse how the projector maps the speech embeddings to embeddings usefull to the LLM. Our research question is: after training, are the projected speech embeddings similar to the text embeddings of the spoken words, or do they occupy a seperate region of the LLM's embedding space that the LLM learns to read? Using the SLAM-ASR method proposed in the paper (frozen WHisper encoder and TinyLLaMA-1.1B-chat, trained on LibriSpeech), we compare each projected speech embeddings to the LLM's text embeddings using cosine similarity, then select the K nearest neighbours to analyse if neighbours align with the spoken words. We will also replace the projected speech embeddings with the nearest neighbour in the LLM's vocabulary to observe the effect on WER. We will also compare our results against an untrained projector for control. Our hypothesis is that the projected speech embeddings do align with the text embeddings of the words spoken.
