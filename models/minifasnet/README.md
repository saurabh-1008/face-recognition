# MiniFASNet model files

Download these files from the Apache-2.0 licensed
[`facenox/face-antispoof-onnx` models directory](https://github.com/facenox/face-antispoof-onnx/tree/main/models):

| File | SHA-256 |
| --- | --- |
| `best_model_quantized.onnx` | `fde20585635cae62ed1d41796f76b6f8bc4b92cd91ec1cf0f1bc6485d2d587a9` |
| `detector_quantized.onnx` | `49f000ec501fef24739071fc7e68267d32209045b6822c0c72dce1da25726f10` |

The binary model files are excluded from this repository. The first model is the quantized
MiniFASNet V2 SE real/spoof classifier; the second is the YuNet face detector used to create its
input crop.
