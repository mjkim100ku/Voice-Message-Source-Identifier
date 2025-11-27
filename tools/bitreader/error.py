class ExpGolombDecodeError(Exception):
    """Represents an error in decoding a number represented in exponential-golomb coding."""
    def __init__(self):
        super().__init__("could not complete Exp-Golomb decode")


class ReaderOutOfBoundsError(Exception):
    """Returned if an attempt is made to read from a parser that has been depleted."""
    def __init__(self):
        super().__init__("cannot read past end of reader bytes")
