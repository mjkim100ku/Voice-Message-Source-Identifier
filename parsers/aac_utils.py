import math

def minInt(x: int, y: int) -> int:
    if x < y:
        return x
    return y

def maxInt(x: int, y: int) -> int:
    if x > y:
        return x
    return y

def aacRound(x: float) -> int:
    return int(math.floor(x + 0.5))
