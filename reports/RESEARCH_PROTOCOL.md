# My baseline and model comparison protocol

My objective is to evaluate a simple image-based identity verification baseline for my CDAC m-governance research. I use InsightFace to extract face embeddings and a JSON file to store user registrations.

## Models I am comparing

| Pack | Published package size (MB) | Recognition backbone |
| --- | ---: | --- |
| buffalo_l | 326 | ResNet50 |
| buffalo_m | 313 | ResNet50 |
| buffalo_s | 159 | MobileFaceNet |
| buffalo_sc | 16 | MobileFaceNet |

I obtained these package sizes and model descriptions from the [official InsightFace model documentation](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md). I initially noted the size of `buffalo_l` as approximately 329 MB. For my local comparison, I measure the extracted ONNX file sizes in decimal MB. These measurements differ from published package sizes and compressed ZIP sizes.

The documentation reports the same recognition accuracy within the `buffalo_l`/`buffalo_m` and `buffalo_s`/`buffalo_sc` groups. Their detectors and auxiliary models differ, so I still need to measure performance on my own evaluation data. My pipeline loads only detection and recognition, although the full downloaded packs can include unused auxiliary files.

I will select a model based on verification errors, latency and storage requirements. Model size and identical-image similarity alone are insufficient for this decision.

## Work I have completed

1. I recorded my initial negative similarity issue and preserved the original scripts under `baseline/`.
2. My corrected pipeline stores model-specific embeddings in JSON and uses the same model for registration and verification.
3. I ran `sanity_pairs.csv` for all four model packs. Each returned a similarity close to 1 for the identical image pair. I use this result to check pipeline consistency, not to estimate recognition accuracy.
4. I saved the local measurements in [SANITY_RESULTS.md](SANITY_RESULTS.md) and the detailed comparison JSON.

## My planned evaluation

1. I will collect consented images of multiple people across sessions, lighting conditions and poses. I will label same-person pairs as `1` and different-person pairs as `0`, include difficult negative pairs, and record the dataset counts and capture conditions.
2. I will keep the people in my development and evaluation sets separate. I will choose thresholds using development pairs and freeze them before evaluation. Every model will use the same evaluation pairs.
3. I will run `compare.py` with CPU execution, a detection size of `640 x 640` and a declared threshold. I will report valid and failed pairs, true positives, true negatives, false positives, false negatives, accuracy, false accept rate and false reject rate.
4. I will report detection/extraction failures separately because the current classification metrics exclude failed pairs. I will consider successful processing coverage alongside those metrics.
5. I will compare measured ONNX size and mean extraction time. My current timing includes image decoding, detection, alignment and embedding extraction after one warm-up. Model loading and downloading are measured separately. I will repeat latency measurements on the target hardware before drawing deployment conclusions.
6. I will explain the tradeoffs between error rates, latency and storage before recommending a model for my project.

## Current limitations

I have completed a baseline pipeline check, but I have not yet completed a held-out verification evaluation or calibrated the threshold. I cannot claim production readiness from the current results. Liveness testing and deployment validation remain outside this experiment.

According to the official documentation, the pretrained models are restricted to non-commercial research. I keep my raw face images and working JSON database excluded from Git, and I will review the source and aggregate reports before sharing them for review.
