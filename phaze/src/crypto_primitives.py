import numpy as np

class RabinFingerprint:
    """A toy implementation of Rabin Fingerprinting for polynomial hashing.
    This simplified version uses direct polynomial evaluation for demonstration purposes.
    """
    def __init__(self, field_size, degree):
        self.field_size = field_size
        self.degree = degree

    def compute_hash(self, data_vector, challenge_point=None):
        """Computes the Rabin fingerprint (hash) of a data vector.

        Args:
            data_vector (list): The input data vector (coefficients of the polynomial).
            challenge_point (int, optional): The point at which to evaluate the polynomial.
                                               If None, a random point within the field is chosen.

        Returns:
            int: The computed hash value.
        """
        if len(data_vector) > self.degree + 1:
            raise ValueError("Data vector length exceeds polynomial degree + 1")

        if challenge_point is None:
            # In a real scenario, this would be a random element from the finite field
            challenge_point = np.random.randint(0, self.field_size)

        hash_value = 0
        for i, coeff in enumerate(data_vector):
            hash_value = (hash_value + coeff * (challenge_point ** i)) % self.field_size
            
        return hash_value


