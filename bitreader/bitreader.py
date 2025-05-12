import math

class BitReader:
    netmasks = [
        0x01,  # 0000 0001
        0x02,  # 0000 0010
        0x04,  # 0000 0100
        0x08,  # 0000 1000
        0x10,  # 0001 0000
        0x20,  # 0010 0000
        0x40,  # 0100 0000
        0x80,  # 1000 0000
    ]

    def __init__(self, input_bytes):
        self.bytes = input_bytes
        self.currentByte = input_bytes[0] if input_bytes else 0
        self.currentByteIndex = 0
        self.posInCurrentByte = 7
        self.length = len(input_bytes)

    # Return the total number of bits left in the stream
    def BitsLeft(self):
        return ((self.BytesLeft() - 1) * 8) + (self.posInCurrentByte + 1)

    # Return the number of bytes left (even if partially read)
    def BytesLeft(self):
        bytes_left = self.length - self.currentByteIndex
        return max(0, bytes_left)

    def ReadBitAsBool(self):
        val, err = self.ReadBit()
        if err:
            return False, err
        if val == 0:
            return False, None
        return True, None

    # Return the number of bits as an unsigned integer
    def ReadBitsAsUInt(self, n):
        result = 0
        for i in range(n):
            bit, err = self.ReadBit()
            if err:
                return 0, err
            result += bit << (n - i - 1)
        return result, None

    # Return the number of bits as an unsigned integer
    def ReadBitsAsUInt8(self, n):
        result = 0
        for i in range(n):
            bit, err = self.ReadBit()
            if err:
                return 0, err
            result += bit << (n - i - 1)
        return result, None

    # Return the number of bits as an unsigned integer
    def ReadBitsAsUInt32(self, n):
        result = 0
        for i in range(n):
            bit, err = self.ReadBit()
            if err:
                return 0, err
            result += bit << (n - i - 1)
        return result, None

    # Return the number of bits as an unsigned integer
    def ReadBitsAsUInt16(self, n):
        result = 0
        for i in range(n):
            bit, err = self.ReadBit()
            if err:
                return 0, err
            result += bit << (n - i - 1)
        return result, None

    # Return the number of bits as a signed integer
    def ReadBitsAsInt(self, n):
        result = 0
        for i in range(n):
            bit, err = self.ReadBit()
            if err:
                return 0, err
            result += bit << (n - i - 1)
        return result, None

    # Return n number of bits into a byte array
    def ReadBitsToByteArray(self, n):
        result = bytearray(math.ceil(n / 8))

        temp = bytearray(n)
        for i in range(n):
            bit, err = self.ReadBit()
            if err:
                return None, err
            temp[i] = bit

        bitmask = 0
        for i in range(n, 0, -1):
            index = len(result) - (bitmask // 8) - 1
            result[index] |= temp[i - 1] << (bitmask % 8)
            bitmask += 1

        return result, None

    # Return n number of bits
    def ReadBits(self, n):
        if n > 8:
            print("ReadBits(n) can only handle up to 8 bits, use ReadBitsToByteArray(n)")

        r = 0

        for i in range(n, 0, -1):
            r <<= 1
            bit, err = self.ReadBit()
            if err:
                return 0, err
            r |= bit
        return r, None

    # Return the next bit from the buffer
    def ReadBit(self):
        if self.BitsLeft() == 0:
            return 0, EOFError("Not enough bits left to read")
        r = (self.currentByte & self.netmasks[self.posInCurrentByte]) >> self.posInCurrentByte
        self.SkipBits(1)
        return r, None

    # Return n number of bytes read from the buffer
    def ReadBytes(self, n):
        arr = bytearray(n)

        if self.BytesLeft() < n:
            return None, EOFError("Not enough bytes left to read")
        else:
            arr[:] = self.bytes[self.currentByteIndex:self.currentByteIndex + n]
            self.currentByteIndex += n
            self.posInCurrentByte = 7

        return arr, None

    # Read Unsigned Exp-Golomb
    def ReadUE(self):
        zeros = 0
        val = 0

        # Count leading zeros
        while True:
            bit, err = self.ReadBit()
            if err:
                return 0, err
            if bit == 0:
                zeros += 1
            else:
                break

        if zeros == 0:
            return 0, None
        else:
            val = 1

        # Shift bits
        while True:
            bit, err = self.ReadBit()
            if err:
                return 0, err
            val <<= 1
            val |= bit
            zeros -= 1
            if zeros == 0:
                return val - 1, None  # Subtract one because we stole a bit for 0.

        # Should not get here
        return 0, Exception("Exp-Golomb decode error")

    # Read Signed Exp-Golomb
    def ReadSE(self):
        u, err = self.ReadUE()
        if err:
            return 0, err
        s = int(u)
        sign = ((s & 0x1) << 1) - 1
        return ((s >> 1) + (s & 0x1)) * sign, None

    # Return n number of bits from the buffer, do not adv. the cursor
    def PeekBits(self, n):
        r = 0

        currentByte = self.currentByte
        posInCurrentByte = self.posInCurrentByte
        currentByteIndex = self.currentByteIndex
        length = self.length

        for i in range(n, 0, -1):
            r <<= 1
            bit = (currentByte & self.netmasks[posInCurrentByte]) >> posInCurrentByte
            r |= bit

            if posInCurrentByte > 0:
                posInCurrentByte -= 1
            else:
                currentByteIndex += 1
                if currentByteIndex <= length - 1:
                    currentByte = self.bytes[currentByteIndex]
                else:
                    currentByte = 0
                posInCurrentByte = 7

        return r, None

    # Return the next bit from the buffer, do not adv. the cursor
    def PeekBit(self):
        if self.BitsLeft() == 0:
            return 0, EOFError("Not enough bits left to read")
        r = (self.currentByte & self.netmasks[self.posInCurrentByte]) >> self.posInCurrentByte
        return r, None

    # Skip n number of bits in the buffer
    def SkipBits(self, n):
        if self.BitsLeft() < n:
            return EOFError("Not enough bits left to skip")
        else:
            for i in range(n):
                if self.posInCurrentByte > 0:
                    self.posInCurrentByte -= 1
                else:
                    self.currentByteIndex += 1
                    if self.currentByteIndex <= self.length - 1:
                        self.currentByte = self.bytes[self.currentByteIndex]
                    else:
                        self.currentByte = 0
                    self.posInCurrentByte = 7
        return None

    # Skip n number of bytes in the buffer
    def SkipBytes(self, n):
        if self.BytesLeft() < n:
            return EOFError("Not enough bytes left to skip")
        else:
            self.currentByteIndex += n
            self.posInCurrentByte = 7
            if self.currentByteIndex < self.length:
                self.currentByte = self.bytes[self.currentByteIndex]
            else:
                self.currentByte = 0
        return None

    # Return if there's a bit left in the stream
    def HasBitLeft(self):
        return self.BitsLeft() > 0

    # Return if there is a byte left in the stream
    def HasByteLeft(self):
        return self.HasBytesLeft(0)

    # Return if there are n bytes left in the stream
    def HasBytesLeft(self, n):
        return self.BytesLeft() > n

    # Reset the stream reader back to the start of the buffer
    def Reset(self):
        self.currentByteIndex = 0
        self.currentByte = self.bytes[0]
        self.posInCurrentByte = 7

    # Perform byte alignment (skip any remaining bits of current byte)
    def ByteAlign(self):
        if self.posInCurrentByte != 7:
            self.SkipBits(self.posInCurrentByte + 1)

    def ByteOffset(self):
        return self.currentByteIndex
