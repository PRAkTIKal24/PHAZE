#![no_std]
extern crate alloc;

use alloc::{vec, vec::Vec};
use risc0_zkvm::guest::env;
use serde::{Deserialize, Serialize};

/// Structure to hold model weights for a simple neural network
#[derive(Serialize, Deserialize)]
pub struct SimpleModelWeights {
    // Simple 2-layer neural network weights
    fc1_weights: Vec<Vec<f32>>,
    fc1_bias: Vec<f32>,
    fc2_weights: Vec<Vec<f32>>,
    fc2_bias: Vec<f32>,
}

/// Structure for the combined input from the host
#[derive(Serialize, Deserialize)]
pub struct ModelInput {
    /// Input tensor for inference
    input_tensor: Vec<f32>,
    
    /// Model weights
    weights: SimpleModelWeights,
}

/// Structure for the model output
#[derive(Serialize, Deserialize)]
pub struct ModelOutput {
    /// Output tensor from inference
    output_tensor: Vec<f32>,
}

/// Apply ReLU activation function
fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Perform matrix-vector multiplication with a bias
fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; weights.len()];
    
    // Compute matrix-vector product: output = weights * input + bias
    for i in 0..weights.len() {
        let mut sum = bias[i];
        for j in 0..input.len() {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }
    
    output
}

/// Perform forward pass through the simple neural network
fn forward(input: &[f32], weights: &SimpleModelWeights) -> Vec<f32> {
    // First layer: Linear + ReLU
    let hidden = linear(input, &weights.fc1_weights, &weights.fc1_bias);
    let hidden_activated = hidden.iter().map(|&x| relu(x)).collect::<Vec<_>>();
    
    // Second layer: Linear (no activation for output)
    let output = linear(&hidden_activated, &weights.fc2_weights, &weights.fc2_bias);
    
    output
}

/// Main entry point for the guest program
pub fn main() {
    // Read the input from the host
    let input: ModelInput = env::read();
    
    // Perform the forward pass
    let output_tensor = forward(&input.input_tensor, &input.weights);
    
    // Create the output structure
    let output = ModelOutput { output_tensor };
    
    // Commit the output to the journal
    env::commit(&output);
}