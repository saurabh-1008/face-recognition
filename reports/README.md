# My research reports

I have organized my findings here so the evaluation can be reviewed without navigating the code or raw data folders.

## Suggested reading order

1. **[Image and identity evaluation](IMAGE_EVALUATION.md)** ? my seven-image test, six confirmed same-person pairs, individual scores and success/failure graphs.
2. **[Observed model limitations](ISSUES.md)** ? limitations supported by my experiments.
3. **[Research protocol](RESEARCH_PROTOCOL.md)** ? my method, evaluation scope and remaining work.
4. **[Sanity results](SANITY_RESULTS.md)** ? my initial identical-image pipeline checks, model sizes and preliminary timings.
5. **[Project structure](PROJECT_STRUCTURE.md)** ? a guide to the supporting files.

## Supporting evidence

| Artifact | Contents |
| --- | --- |
| [Evaluation JSON](../results/image_evaluation.json) | Per-image outcomes, pair scores, settings and software versions |
| [Evaluation CSV](../results/image_evaluation.csv) | Per-image successes, failures and timings |
| [Extraction graph](../results/graphs/image_evaluation.png) | Success/failure matrix across images and models |
| [Same-person graph](../results/graphs/genuine_pairs.png) | Six confirmed genuine pair scores at threshold 0.5 |
| [Initial comparison graph](../results/graphs/model_comparison.png) | Storage, preliminary latency and same-image similarity |
| [Image labels](../experiments/image_labels.csv) | Confirmed person grouping; unknown identities remain blank |

My current identity evaluation covers one confirmed person. I distinguish extraction success from a correct identity decision and have not yet measured false accept rate on confirmed different-person pairs.

[Back to the project overview](../README.md) ? [Setup and commands](../docs/RUNNING.md)
