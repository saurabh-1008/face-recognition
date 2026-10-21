# TinyLiveness EfficientNet-B0

Source pinned to revision `8d92198e7ee122f39b2585e691a3d9450ca0cb0f`:
https://github.com/yuvrajraina/TinyLiveness/tree/8d92198e7ee122f39b2585e691a3d9450ca0cb0f

- ONNX: `tinyliveness_main_apcer1_224.onnx` (16,039,289 bytes)
- SHA-256: `fbf9258f27b6b597c8fc8767a1add930628fe90b396126b15171b7f54985c670`
- Input: NCHW RGB float32, 224px, ImageNet mean/std normalization.
- Output: one live probability; default release threshold 0.99.
- The upstream MIT license and exact release policy are preserved alongside this README.

Use the common runner with `--model tinyliveness`. The adapter supplies a YuNet five-landmark
similarity alignment to a 224px ArcFace template. This is our chosen alignment implementation,
not evidence of reproducing the publisher's benchmark. See ../../docs/LIVENESS_TESTING.md.
