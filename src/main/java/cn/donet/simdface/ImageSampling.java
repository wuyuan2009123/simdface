package cn.donet.simdface;

/** Bilinear sampling from tightly packed RGB24 into a channels-first tensor. */
final class ImageSampling {

    private ImageSampling() {
    }

    static void validate(byte[] rgb, int width, int height) {
        if (width <= 0 || height <= 0 || (long) width * height != rgb.length / 3
                || rgb.length % 3 != 0) {
            throw new IllegalArgumentException("Expected positive dimensions and tightly packed RGB24 pixels.");
        }
    }

    static void writePixel(byte[] rgb, int width, int height, float x, float y,
            Tensor target, int index, boolean bgr) {
        if (!(x > -1 && x < width && y > -1 && y < height)) {
            return;
        }
        int left = (int) Math.floor(x), top = (int) Math.floor(y);
        float fx = x - left, fy = y - top;
        int upperLeft = pixelOffset(width, height, left, top);
        int upperRight = pixelOffset(width, height, left + 1, top);
        int lowerLeft = pixelOffset(width, height, left, top + 1);
        int lowerRight = pixelOffset(width, height, left + 1, top + 1);
        for (int c = 0; c < 3; c++) {
            int channel = bgr ? 2 - c : c;
            float upper = sample(rgb, upperLeft, channel) * (1 - fx)
                    + sample(rgb, upperRight, channel) * fx;
            float lower = sample(rgb, lowerLeft, channel) * (1 - fx)
                    + sample(rgb, lowerRight, channel) * fx;
            target.data()[c * target.plane() + index] = (float) Math.rint(upper * (1 - fy) + lower * fy);
        }
    }

    private static int pixelOffset(int width, int height, int x, int y) {
        return Integer.compareUnsigned(x, width) < 0 && Integer.compareUnsigned(y, height) < 0
                ? (y * width + x) * 3 : -1;
    }

    private static int sample(byte[] rgb, int offset, int channel) {
        return offset < 0 ? 0 : rgb[offset + channel] & 0xFF;
    }
}
