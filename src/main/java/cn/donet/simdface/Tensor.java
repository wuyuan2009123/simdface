package cn.donet.simdface;

/** Channels-first tensor with tightly packed float data. */
record Tensor(int channels, int height, int width, float[] data) {

    int plane() {
        return height * width;
    }

    Tensor withData(float[] newData) {
        return new Tensor(channels, height, width, newData);
    }
}
