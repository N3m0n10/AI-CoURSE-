import numpy as np
L = 20

def get_bits(x):
    """Convert a number between 0 and π to binary string"""

    # Normalize x to be between 0 and 1
    normalized = x / np.pi
    # Convert to integer between 0 and 2^L-1
    int_value = int(normalized * ((2**L) - 1))
    # Convert to binary and remove '0b' prefix
    binary = bin(int_value)[2:]
    # Pad with zeros to reach length L
    return binary.zfill(L)

def get_float(bits):
    """Convert binary string back to number between 0 and π"""
    # Convert binary string to integer
    int_value = int(bits, 2)
    # Convert to value between 0 and 1
    normalized = int_value / ((2**L) - 1)
    # Scale to range [0, π]
    return normalized * np.pi

a = get_bits(2.37)
print(a)
print(get_float(a))
