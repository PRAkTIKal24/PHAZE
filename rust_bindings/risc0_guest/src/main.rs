#![no_main]

use risc0_guest::main;

risc0_zkvm::guest::entry!(main);