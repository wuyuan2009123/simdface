package cn.donet.simdface;

import java.nio.charset.StandardCharsets;

/**
 * Minimal protobuf wire-format reader over a shared byte array. Values are exposed as
 * (offset, length) slices so nested messages do not copy weight blobs.
 */
final class ProtoReader {

    private final byte[] buffer;
    private final int limit;
    private int position;

    int field;
    long number;
    int valueOffset;
    int valueLength;

    ProtoReader(byte[] buffer) {
        this(buffer, 0, buffer.length);
    }

    ProtoReader(byte[] buffer, int start, int end) {
        if (start < 0 || end > buffer.length || start > end) {
            throw new IllegalArgumentException("Invalid protobuf slice.");
        }
        this.buffer = buffer;
        this.position = start;
        this.limit = end;
    }

    boolean atEnd() {
        return position == limit;
    }

    boolean next() {
        if (atEnd()) {
            return false;
        }
        long tag = readVarint();
        field = (int) (tag >>> 3);
        int wireType = (int) (tag & 7);
        number = 0;
        valueOffset = position;
        valueLength = 0;
        switch (wireType) {
            case 0 -> number = readVarint();
            case 1 -> take(8);
            case 2 -> take((int) readVarint());
            case 5 -> take(4);
            default -> throw new InvalidDataException("Unsupported ONNX protobuf wire type.");
        }
        return true;
    }

    long readVarint() {
        long result = 0;
        for (int shift = 0; shift < 64 && position < limit; shift += 7) {
            int next = buffer[position++] & 0xFF;
            result |= (long) (next & 0x7F) << shift;
            if (next < 0x80) {
                return result;
            }
        }
        throw new InvalidDataException("Invalid ONNX protobuf varint.");
    }

    private void take(int length) {
        if (length < 0 || length > limit - position) {
            throw new InvalidDataException("Truncated ONNX protobuf field.");
        }
        valueOffset = position;
        valueLength = length;
        position += length;
    }

    ProtoReader slice() {
        return new ProtoReader(buffer, valueOffset, valueOffset + valueLength);
    }

    String utf8() {
        return new String(buffer, valueOffset, valueLength, StandardCharsets.UTF_8);
    }

    float floatLE() {
        return Float.intBitsToFloat(intLE(valueOffset));
    }

    int intLE(int offset) {
        return (buffer[offset] & 0xFF)
                | (buffer[offset + 1] & 0xFF) << 8
                | (buffer[offset + 2] & 0xFF) << 16
                | (buffer[offset + 3] & 0xFF) << 24;
    }

    long uintLE(int offset) {
        return (long) (buffer[offset] & 0xFF)
                | (long) (buffer[offset + 1] & 0xFF) << 8
                | (long) (buffer[offset + 2] & 0xFF) << 16
                | (long) (buffer[offset + 3] & 0xFF) << 24
                | (long) (buffer[offset + 4] & 0xFF) << 32
                | (long) (buffer[offset + 5] & 0xFF) << 40
                | (long) (buffer[offset + 6] & 0xFF) << 48
                | (long) (buffer[offset + 7] & 0xFF) << 56;
    }

    int valueEnd() {
        return valueOffset + valueLength;
    }
}
