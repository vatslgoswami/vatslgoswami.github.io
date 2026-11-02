---
title: ArtEmis Image Captioning
year: 2025
kind: Project
summary: Generating descriptions of artworks using CNN–LSTM and transformer-based image captioning models.
tags: [Computer Vision, NLP, Deep Learning]
links:
  - { label: GitHub, url: "https://github.com/roshnipai05/ArtEmis" }
detail: true
---

- Trained image captioning models on a subset of the ArtEmis dataset, which pairs WikiArt paintings with human-written descriptions of the emotions and interpretations they evoke. The project explored both a custom CNN–LSTM architecture and a vision-transformer encoder with a transformer decoder.
- Built image and text preprocessing pipelines for caption generation, including vocabulary construction, subword tokenization, and pretrained word embeddings. The CNN–LSTM model injects image features into the decoder's initial state; the transformer model uses image patch features to condition text generation.
