# Categorical sampling with TensorRT plugin

## Overview
Since TensorRT does not support categorical sampling, we add that as a **TensorRT Plugin**.

To include the operation in a trace model, the user can refer to the plugin by name in the PyTorch code being traced. During tracing that operation becomes part of the ONNX graph. When TensorRT converts the ONNX graph to TensorRT it encounters this operation and looks for the corresponding name among its plugin. If we have provided the plugin to TensorRT it can find it and incorporate it in the engine it builds.

There are few pieces to discuss:

* Categorical sampling CUDA Kernel: implements categorical sampling on the GPU.

* Categorcial sampling TensorRT plugin (C++): wraps the CUDA kernel as a TensorRT plugin, implementing the TensorRT plugin APIs:
`IPluginV3`, `IPluginV3OneCore`, `IPluginV3OneBuild` and `IPluginV3OneRuntime`.

* `linear_lt_autoregressive.ipynb`: Demonstrates of how start from PyTorch code that uses the `CategoricalSampling` operation and trace it to ONNX and then TensorRT. It has an example of an autoregressive loop running a dummy LT (embedding layer + linear) and sampling at each step, all traced into a single graph via PyTorch -> ONNX -> TensorRT.

## File structure
```
tensorrt_llm/  (top-level)
├── linear_lt_autoregressive.ipynb      # demo notebook showing how to trace a model with the custom sampling operation
└── categorical_sampling_plugin/
    ├── README_plugin.md                # This file
    ├── build_plugin.sh                 # Builds the plugin
    ├── build_kernel_test.sh            # (optional) Builds the kernel unit test.
    ├── CMakeLists.txt                  # For building the plugin
    ├── CategoricalSamplingPlugin.cpp   # Plugin implementation
    ├── CategoricalSamplingPlugin.h     # Plugin header
    └── cuda_kernel/                    # Kernel subdirectory
        ├── CMakeLists.txt              # Kernel unit test build
        ├── include/                     # Kernel header
        │   └── categorical_sampling.cuh
        ├── src/                         # Kernel implementation
        │   └── categorical_sampling.cu
        └── test/                        # Kernel unit test
            └── test_categorical_sampling_fp16.cu
```
## How to build and use the plugin

### 1. Build the plugin

```bash
cd categorical_sampling_plugin
./build_plugin.sh
```
This will build both the plugin (including the CUDA kernel) and copy the resulting shared library to:
`tensorrt_llm/build/libcategorical_sampling_plugin.so`.

Note that there is also a second script (`build_kernel_test.sh`) that builds a standalone unit test for the CUDA kernel. But you don't need that to use the plugin.

### 2. Define Categorical Sampling as a custom operation in your PyTorch code

```python
import torch
from torch import Tensor

class CategoricalSamplingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        # Forward is only used in eager runs (not during ONNX parsing).
        # So you can put a dummy implementation here.
        # We return a 1D INT32 tensor to match plugin output type
        return torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)

    @staticmethod
    def symbolic(g, x):
        # Emit ONNX node whose (domain, op_type) matches the TRT plugin creator.
        output = g.op("CategoricalSampling", x)
        # Set the output type to INT32 with 1D shape (dynamic size)
        output.setType(x.type().with_dtype(torch.int32).with_sizes([None]))
        return output

# IMPORTANT: This function is what you call in your model. See `linear_lt_autoregressive.ipynb` for an example.
def categorical_sampling(x: Tensor) -> Tensor:
    return CategoricalSamplingFn.apply(x)
```

### 3. Trace the model to ONNX
Use `torch.export`. See `linear_lt_autoregressive.ipynb` for an example.

### 4. Convert to TensorRT and run
```bash
/usr/local/tensorrt/targets/x86_64-linux-gnu/bin/trtexec --fp16 \
    --onnx=/code/tensorrt_llm/local_transformer.onnx \
    --saveEngine=/code/tensorrt_llm/local_transformer.trt \
    --minShapes=hidden_states:2x768  \
    --optShapes=hidden_states:8x768 \
    --maxShapes=hidden_states:32x768 \
    --plugins=/code/tensorrt_llm/build/libcategorical_sampling_plugin.so
```
The last line points TensorRT to our plugin.

# Notes
The plugin is built as a shared library (`*.so`). It appears that `trtexec` loads the library dynamically during execution but does **not** incorporate it into the engine itself. We will need to figure out how this loading works when executing from the TRT-LLM runtime or alternatively try to statically link it into the `TRT-LLM` runtime.
