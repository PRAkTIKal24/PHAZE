import rust_zkml_bindings

result_sum = rust_zkml_bindings.sum_as_string(5, 3)
print(f"Result from Rust sum: {result_sum}")

data_to_hash = b"hello world"
result_hash = rust_zkml_bindings.hash_data(list(data_to_hash))
print(f"Result from Rust hash: {result_hash}")


