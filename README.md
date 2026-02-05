## Project Overview

This project aims to be an extended version from my senior capstone project. During trade study process of selecting machine learning method for 3D metal defect detection process, our team chose Convolutional Neural Network + Transfer Learning to achieve the highest accuracy of defect detection. While it was based on comparing and referencing previous studies, I realized that it's not enough to evaluate the best Machine learning method for space system. Overall, this project approaches from software engineering perspective rather than aerospace engineering.

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
tensor([[0.6627, 0.5539, 0.4431],
        [0.4663, 0.2183, 0.0328],
        [0.6295, 0.8436, 0.8391],
```
