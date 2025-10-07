import numpy as np
import pytest

from phaze import RabinFingerprint


def test_rabin_fingerprint_init():
    fingerprinter = RabinFingerprint(field_size=101, degree=5)
    assert fingerprinter.field_size == 101
    assert fingerprinter.degree == 5


def test_rabin_fingerprint_compute_hash_valid_input():
    fingerprinter = RabinFingerprint(field_size=2**31 - 1, degree=10)
    data_vector = [np.random.randint(0, 100) for _ in range(10)]
    hash_value = fingerprinter.compute_hash(data_vector)
    assert isinstance(hash_value, int)


def test_rabin_fingerprint_compute_hash_invalid_input_length():
    fingerprinter = RabinFingerprint(field_size=101, degree=5)
    data_vector = [1, 2, 3, 4, 5, 6, 7]  # Length > degree + 1 (7 > 6)
    with pytest.raises(
        ValueError, match=r"Data vector length exceeds polynomial degree \+ 1"
    ):
        fingerprinter.compute_hash(data_vector)


def test_rabin_fingerprint_compute_hash_zero_vector():
    fingerprinter = RabinFingerprint(field_size=101, degree=5)
    data_vector = [0, 0, 0, 0, 0, 0]
    hash_value = fingerprinter.compute_hash(data_vector)
    assert hash_value == 0


def test_rabin_fingerprint_compute_hash_large_values():
    fingerprinter = RabinFingerprint(field_size=2**31 - 1, degree=10)
    data_vector = [np.random.randint(0, 2**30) for _ in range(10)]
    hash_value = fingerprinter.compute_hash(data_vector)
    assert isinstance(hash_value, int)


def test_rabin_fingerprint_compute_hash_consistency():
    fingerprinter = RabinFingerprint(field_size=101, degree=5)
    data_vector = [1, 2, 3, 4, 5]
    hash1 = fingerprinter.compute_hash(data_vector, challenge_point=7)
    hash2 = fingerprinter.compute_hash(data_vector, challenge_point=7)
    assert hash1 == hash2
