import numpy as np 
import math

def _first_primes(count):
    primes = []
    candidate = 2

    while len(primes) < count:
        is_prime = True

        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break

        if is_prime:
            primes.append(candidate)

        candidate += 1

    return primes


def _halton_sequence(count, dim, start_index=1):
    bases = _first_primes(dim)
    sequence = np.empty((count, dim), dtype=float)

    for row in range(count):
        index = start_index + row

        for column, base in enumerate(bases):
            value = 0.0
            factor = 1.0
            number = index

            while number > 0:
                factor /= base
                value += factor * (number % base)
                number //= base

            sequence[row, column] = value

    return sequence


def _levy_flight(rng, size, exponent=1.5):
    sigma = (
        math.gamma(1.0 + exponent)
        * np.sin(np.pi * exponent / 2.0)
        / (
            math.gamma((1.0 + exponent) / 2.0)
            * exponent
            * 2.0 ** ((exponent - 1.0) / 2.0)
        )
    ) ** (1.0 / exponent)

    u = rng.normal(0.0, sigma, size=size)
    v = rng.normal(0.0, 1.0, size=size)

    return u / np.maximum(np.abs(v), np.finfo(float).eps) ** (1.0 / exponent)
