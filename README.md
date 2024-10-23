# ACT: Action Chunking with Transformers on tinygrad

This repository contains an implementation of ACT (Action Chunking with Transformers) using tinygrad, a lightweight deep learning framework.

An overview of ACT with ALOHA (low cost bimanipulation hardware): https://tonyzhaozh.github.io/aloha/aloha.pdf

## Features
* Implementation of ACT model architecture using tinygrad
* Support for simulated robotic manipulation tasks
* Training and evaluation scripts
* Integration with tinygrad's lazy evaluation and JIT compilation

## How to use

### Training

```
BEAM=2 DEBUG=2 python3.10 train.py
```

### Eval

```
DEBUG=2 MUJOCO_GL=glfw python3.10 test.py
```
