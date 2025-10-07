import secrets
from typing import List, Tuple

import numpy as np


class RabinFingerprint:
    """A toy implementation of Rabin Fingerprinting for polynomial hashing.
    This simplified version uses direct polynomial evaluation for demonstration.
    """

    def __init__(self, field_size=2**31 - 1, degree=100):
        self.field_size = field_size
        self.degree = degree

    def compute_hash(self, data_vector, challenge_point=None):
        """Computes the Rabin fingerprint (hash) of a data vector.

        Args:
            data_vector (list or bytes): The input data vector (coefficients).
            challenge_point (int, optional): Point to evaluate polynomial.
                If None, a random point within the field is chosen.

        Returns:
            int: The computed hash value.
        """
        # Handle bytes input
        if isinstance(data_vector, bytes):
            data_vector = list(data_vector)

        if len(data_vector) > self.degree + 1:
            raise ValueError("Data vector length exceeds polynomial degree + 1")

        if challenge_point is None:
            # In a real scenario, this would be a random element from the finite field
            challenge_point = np.random.randint(1, self.field_size)

        hash_value = 0
        for i, coeff in enumerate(data_vector):
            hash_value = (hash_value + coeff * (challenge_point**i)) % self.field_size

        return hash_value


class ShamirSecretSharing:
    """A simplified implementation of Shamir's Secret Sharing scheme."""

    def __init__(self, threshold: int, num_shares: int, prime: int = 2**31 - 1):
        """Initialize Shamir Secret Sharing.

        Args:
            threshold: Minimum number of shares needed to reconstruct the secret
            num_shares: Total number of shares to generate
            prime: Prime number for finite field arithmetic
        """
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")

        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = prime

    def _mod_inverse(self, a: int, m: int) -> int:
        """Compute modular inverse using extended Euclidean algorithm."""
        if a < 0:
            a = (a % m + m) % m

        # Extended Euclidean Algorithm
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method."""
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.prime
        return result

    def generate_shares(self, secret: bytes) -> List[Tuple[int, bytes]]:
        """Generate shares for the secret.

        Args:
            secret: The secret to be shared (as bytes)

        Returns:
            List of (x, y) tuples representing the shares
        """
        # Convert secret bytes to integers
        secret_ints = list(secret)
        shares = []

        for byte_idx, secret_byte in enumerate(secret_ints):
            # Generate random coefficients for polynomial
            coefficients = [secret_byte]  # a0 = secret
            for _ in range(self.threshold - 1):
                coefficients.append(secrets.randbelow(self.prime))

            # Generate shares by evaluating polynomial at different points
            byte_shares = []
            for i in range(1, self.num_shares + 1):
                y = self._evaluate_polynomial(coefficients, i)
                byte_shares.append((i, y))

            if byte_idx == 0:
                # Initialize shares list
                shares = [(x, [y]) for x, y in byte_shares]
            else:
                # Append to existing shares
                for j, (_x, y) in enumerate(byte_shares):
                    shares[j][1].append(y)

        # Convert y values back to bytes
        final_shares = []
        for x, y_list in shares:
            y_bytes = bytes(y % 256 for y in y_list)  # Ensure values fit in bytes
            final_shares.append((x, y_bytes))

        return final_shares

    def reconstruct_secret(self, shares: List[Tuple[int, bytes]]) -> bytes:
        """Reconstruct the secret from shares using Lagrange interpolation.

        Args:
            shares: List of (x, y) tuples representing the shares

        Returns:
            The reconstructed secret as bytes
        """
        if len(shares) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} shares to reconstruct secret"
            )

        # Use only the first threshold shares
        shares = shares[: self.threshold]

        # Get the length of the secret from the first share
        secret_length = len(shares[0][1])
        reconstructed_bytes = []

        # Reconstruct each byte of the secret
        for byte_idx in range(secret_length):
            # Extract the y values for this byte position
            points = [(x, y_bytes[byte_idx]) for x, y_bytes in shares]

            # Lagrange interpolation to find a0 (the secret byte)
            secret_byte = 0
            for i, (xi, yi) in enumerate(points):
                # Calculate Lagrange basis polynomial Li(0)
                li_0 = 1
                for j, (xj, _) in enumerate(points):
                    if i != j:
                        # Li(0) = product of (-xj) / (xi - xj) for all j != i
                        numerator = (-xj) % self.prime
                        denominator = (xi - xj) % self.prime
                        li_0 = (
                            li_0
                            * numerator
                            * self._mod_inverse(denominator, self.prime)
                        ) % self.prime

                secret_byte = (secret_byte + yi * li_0) % self.prime

            reconstructed_bytes.append(secret_byte % 256)  # Ensure byte range

        return bytes(reconstructed_bytes)
