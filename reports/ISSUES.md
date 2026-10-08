# Model limitations observed in my experiments

I record model limitations here using observations from my experiments. My current results establish a detection failure on one image and the storage requirements of the downloaded model packs. I have not yet established general recognition accuracy or robustness across different capture conditions.

## LIMITATION-001: A visible face was not detected

**Status:** Reproduced with all four Buffalo packs; further evaluation needed.

When I tested `data/images/IMG-20250224-WA0003.jpg`, the image loaded successfully, but the detector returned **0 faces**. I reproduced this using `buffalo_l` with a detection input size of **640 x 640** and the default detection threshold of **0.5**.

The image visibly contains a face. It is a tight close-up with an angled face, which may contribute to the failure, but I have not established the exact cause. I subsequently tested all seven available images with all four packs. Every pack failed to detect a face in this image and successfully extracted embeddings from the other six images. These findings apply to this small dataset and the current configuration. See [IMAGE_EVALUATION.md](IMAGE_EVALUATION.md) for the per-image outcomes.

Because detection failed, I could not extract an embedding or calculate similarity. For my m-governance workflow, this means a submitted image can fail before identity verification begins. I will record these failures separately from incorrect match decisions.

I will test more images across framing, pose and lighting conditions, using the same inputs for each model pack. This will help me measure detection coverage and determine how often this limitation occurs.

## LIMITATION-002: Model packs require local storage

**Status:** Measured locally; deployment impact still needs evaluation.

My downloaded packs occupy the following sizes for their extracted ONNX files:

| Model pack | ONNX file size (decimal MB) |
| --- | ---: |
| buffalo_l | 341.27 |
| buffalo_m | 327.64 |
| buffalo_s | 166.10 |
| buffalo_sc | 16.14 |

I recorded these measurements in [sanity_comparison.json](../results/sanity_comparison.json). These are full-pack ONNX sizes, including auxiliary models where present; my current pipeline loads only detection and recognition. They do not represent runtime memory usage or compressed download size.

The larger packs increase the local storage requirement for my project. I still need to evaluate whether this is a practical constraint on my intended deployment hardware and whether the smaller packs provide acceptable verification performance.

## Scope of my current findings

My identical-image checks succeeded for all four packs, but they do not measure recognition accuracy on different photos. I will report recognition limitations only after evaluating labelled same-person and different-person pairs. My evaluation plan is recorded in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

## LIMITATION-003: Smaller recognition models rejected confirmed same-person pairs

**Status:** Observed on my labelled local dataset at similarity threshold 0.5.

I confirmed that four images show the same person and evaluated their six distinct pairs. `buffalo_l` and `buffalo_m` accepted all six. `buffalo_s` and `buffalo_sc` each accepted four and rejected two.

The rejected pairs were the PNG image (`83ea0662-9151-4ed9-857e-96c9d5fa6a3e.png`) paired with `IMG_20250926_105517.jpg` (similarity approximately 0.467), and the same PNG paired with `SAVE_20260908_223839.jpg` (approximately 0.483). Both scores were below my experimental threshold of 0.5.

These are false rejections under my confirmed labels and current threshold. I have not established the cause or calibrated a better threshold. Lowering the threshold could change the results but needs evaluation with different-person pairs to assess false accepts. One `buffalo_l` genuine score was also close to the threshold at approximately 0.501.

I recorded all pair scores in [IMAGE_EVALUATION.md](IMAGE_EVALUATION.md) and [image_evaluation.json](../results/image_evaluation.json). These findings apply to one confirmed identity and do not establish a general model ranking.
