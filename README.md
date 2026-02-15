## Project Overview

This project aims to be an extended version from my senior capstone project. During trade study process of selecting machine learning method for 3D metal defect detection process, our team chose Convolutional Neural Network + Transfer Learning to achieve the highest accuracy of defect detection. While it was based on comparing and referencing previous studies, I realized that it's not enough to evaluate the best Machine learning method for space system. Overall, this project approaches from software engineering perspective rather than aerospace engineering.

## Data Acquisition

This was also challenging part from trade study during Senior Capstone challenge because there was a no such datasets to be used for pre-trained due to its novelty method of 3D metal manufacturing we selected: DAED (Directed Acoustic Energy Deposition). For this project, I searched public datasets.

## Project Development

### Project Setup

1. Create a project: `uv init mlb`: Acronym for Machine Learning Benchmark
2. Create a python virtual environment: `uv venv`, and activate: `source .venv/bin/activate`
3. Install packages:`uv add torch torchvision`
4. Testing PyTorch

```python
import torch

def main():
    x = torch.rand(5, 3)
    print(x)

if __name__ == "__main__":
    main()

```

```zsh
▶ uv run main.py
Uninstalled 1 package in 906ms
Installed 1 package in 245ms
tensor([[0.6627, 0.5539, 0.4431],
        [0.4663, 0.2183, 0.0328],
        [0.6295, 0.8436, 0.8391],
```

## Step 1: Comparing PyTorch performance on M1 macOS CPU vs GPU

- Goal: Compare CPU vs GPU (MPS) performance using the exact same workload and settings.
- Rationale: This hardware baseline makes later comparisons between CNN, transfer learning, and other methods fair and reproducible.
- Backend: On macOS, PyTorch uses [MPS (Metal Performance Shaders)](https://developer.apple.com/metal/pytorch/) for GPU.
- Metrics:

1. Latency (time per iteration)
2. Throughput (iterations or images per second)
3. Speedup (CPU time / GPU time)
4. Consistency (average across multiple runs)

- Controls (must stay identical across CPU/GPU):

1. Same input size and batch size
2. Same model or operation
3. Same dtype (float32)
4. Same warmup and measured iterations

- Timing rule: GPU work is asynchronous, so synchronize before and after timing (`torch.mps.synchronize()`).
- Conclusion placeholder: If MPS shows lower latency and higher throughput, I will use GPU for the rest of this project.

### Concepts I learned

"Tensor":

    - Shape (`.shape`): Tuple that describes the dimensions (e.g. n x n)
    - Device (`.device`): The place where the tensor lives, CPU or CUDA (GPU)
    - Type (`.dtype`): Data Type of the numbers. The default is float32.

> The reason why it's float32 was intentional because of gradient. The entire of engine of deep learning works by making tiny, continuous adjustments to a model's weights. Model parameters (weights, bias) MUST be a float type. So, float32 is standard.
