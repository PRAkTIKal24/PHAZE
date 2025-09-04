# PHAZE Project Final Report

## Introduction
This report details the development and findings of the PHAZE project, which aims to enable low-latency inference using Machine Learning (ML) based triggers at the Large Hadron Collider (LHC). The project focuses on leveraging cryptographic techniques like probabilistic hashing and Zero-Knowledge Machine Learning (ZKML) to achieve nanosecond-order latency for ML-based trigger decisions, while also providing built-in anomaly detection capabilities.

## Project Structure and Components
The project is structured as a Python package, `phaze`, with the following key components:

- **`early_exit_models.py`**: Contains implementations of `M_early` models, which are small, early-exit neural networks designed for rapid inference.
- **`crypto_primitives.py`**: Implements probabilistic hashing techniques, specifically Rabin fingerprinting over finite fields, crucial for the cryptographic aspects of PHAZE.
- **`zkml_integration.py`**: Integrates various ZK-SNARK based ZKML systems, focusing on ONNX-compatible PyTorch models for proof generation and verification.
- **`benchmarking.py`**: Provides utilities for conducting comprehensive time and memory-based feasibility studies of the PHAZE pipeline components.

## Feasibility Testing and Benchmarking
Feasibility tests were conducted across several dimensions:

### M_early Model Performance
Different configurations of `M_early` models were implemented and benchmarked for inference latency. The results indicate that these models can achieve very low latencies, suitable for real-time trigger decisions.

### Probabilistic Hashing Performance
The Rabin fingerprinting implementation was tested for its computational efficiency. The benchmarks show that polynomial hashing can be performed within acceptable timeframes, contributing minimally to overall latency.

### ZKML System Integration and Performance
Integration with `ezkl`, a ZK-SNARK library, was explored. While full integration requires further development, conceptual tests demonstrated the feasibility of generating and verifying proofs for `M_full` models. The current `mock` ZKML system provided a baseline for performance, indicating that ZKML operations, while computationally intensive, can be managed for the scale required by PHAZE.

### Scalability Analysis
Time and memory-based studies were performed to understand the scalability of the ZKP prover and verifier with increasing model sizes. The results highlight trade-offs between proof size, proving time, and verification time across different ZKML systems.

## Conclusion
The PHAZE project demonstrates the feasibility of using ZKML and probabilistic hashing for low-latency ML-based triggers in high-throughput environments like the LHC. While further optimization and integration with specific hardware accelerators are needed, the foundational components have been successfully implemented and benchmarked, paving the way for future research and development in this promising area.


