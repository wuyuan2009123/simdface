package cn.donet.simdface;

record TensorSlot(int offset, int channels, int height, int width) {

    int length() {
        return Math.multiplyExact(channels, Math.multiplyExact(height, width));
    }
}
