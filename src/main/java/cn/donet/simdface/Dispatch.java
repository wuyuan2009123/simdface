package cn.donet.simdface;

import java.nio.ByteBuffer;

/** Push-constant layout shared with both GLSL compute shaders. */
final class Dispatch {

    static final int SIZE = 84;

    int op;
    int source;
    int destination;
    int weights;
    int bias;
    int auxiliary;
    int extra;
    int inChannels;
    int inHeight;
    int inWidth;
    int outChannels;
    int outHeight;
    int outWidth;
    int kernelHeight;
    int kernelWidth;
    int strideY;
    int strideX;
    int padY;
    int padX;
    int groups;
    float scalar;

    void writePushConstants(ByteBuffer buffer) {
        int offset = 0;
        buffer.putInt(offset, op);
        buffer.putInt(offset += 4, source);
        buffer.putInt(offset += 4, destination);
        buffer.putInt(offset += 4, weights);
        buffer.putInt(offset += 4, bias);
        buffer.putInt(offset += 4, auxiliary);
        buffer.putInt(offset += 4, extra);
        buffer.putInt(offset += 4, inChannels);
        buffer.putInt(offset += 4, inHeight);
        buffer.putInt(offset += 4, inWidth);
        buffer.putInt(offset += 4, outChannels);
        buffer.putInt(offset += 4, outHeight);
        buffer.putInt(offset += 4, outWidth);
        buffer.putInt(offset += 4, kernelHeight);
        buffer.putInt(offset += 4, kernelWidth);
        buffer.putInt(offset += 4, strideY);
        buffer.putInt(offset += 4, strideX);
        buffer.putInt(offset += 4, padY);
        buffer.putInt(offset += 4, padX);
        buffer.putInt(offset += 4, groups);
        buffer.putFloat(offset + 4, scalar);
    }
}
