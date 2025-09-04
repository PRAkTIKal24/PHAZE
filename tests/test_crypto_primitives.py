import pytest
import numpy as np
from phaze.crypto_primitives import RabinFingerprint

def test_rabin_fingerprint_init():
    fingerprinter = RabinFingerprint(field_size=100, degree=5)
    assert fingerprinter.field_size == 100
    assert fingerprinter.degree == 5

def test_rabin_fingerprint_compute_hash_basic():
    fingerprinter = RabinFingerprint(field_size=10007, degree=3)
    data_vector = [1, 2, 3]
    hash_value = fingerprinter.compute_hash(data_vector)
    assert isinstance(hash_value, int)
    assert 0 <= hash_value < 10007

def test_rabin_fingerprint_compute_hash_with_challenge_point():
    fingerprinter = RabinFingerprint(field_size=10007, degree=3)
    data_vector = [1, 2, 3]
    challenge_point = 5.0
    hash_value = fingerprinter.compute_hash(data_vector, challenge_point=challenge_point)
    assert isinstance(hash_value, int)
    assert 0 <= hash_value < 10007

def test_rabin_fingerprint_compute_hash_long_vector_raises_error():
    fingerprinter = RabinFingerprint(field_size=100, degree=1)
    data_vector = [1, 2, 3] # Length 3, degree + 1 = 2
    with pytest.raises(ValueError, match="Data vector length exceeds polynomial degree \+ 1"):
        fingerprinter.compute_hash(data_vector)

def test_rabin_fingerprint_polynomial_from_vector_conceptual():
    fingerprinter = RabinFingerprint(field_size=100, degree=2)
    data_vector = [10, 20, 30]
    poly_func = fingerprinter._polynomial_from_vector(data_vector)
    # Test a few points, conceptually
    assert abs(poly_func(0) - 10) < 1e-9
    assert abs(poly_func(1) - 20) < 1e-9
    assert abs(poly_func(2) - 30) < 1e-9


