Benchmarking Homomorphic Encryption for Neural Network Layers
=============================================================

This repository contains the code and experiments for the project **“Benchmarking Homomorphic Encryption for Neural Network Layers”** by Alsu Giliazova, Maksim Zarkov, and Vahan Yeranosyan (LMU Munich).

The notebooks evaluate the practical feasibility of homomorphic encryption (HE) for neural network inference, focusing on:

- **Vector and linear regression benchmarks** with CKKS (via Pyfhel) to study accuracy, runtime, and scaling.
- **Encrypted MLP on MNIST** (`MLP.ipynb`) using TenSEAL, comparing plaintext vs encrypted inference for fully connected layers.
- **Encrypted CNN on MNIST** (`CNN.ipynb`), analyzing the overhead of homomorphic convolutions versus plaintext execution.

Across these experiments, we benchmark:

- **Runtime and slowdown** of encrypted vs plaintext inference.
- **Memory usage and ciphertext size**.
- **Accuracy degradation** from polynomial activations and CKKS approximation.



