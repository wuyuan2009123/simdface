package cn.donet.simdface;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/** CPU kernels for the operator set used by the pinned YuNet and SFace graphs. */
final class Kernels {

    private static final int THREADS = Math.max(1, Integer.getInteger("simdface.threads",
            Math.min(8, Runtime.getRuntime().availableProcessors())));
    private static final ForkJoinPool POOL = new ForkJoinPool(THREADS);

    private Kernels() {
    }

    private static void forChannels(int count, IntConsumer body) {
        if (THREADS == 1 || count <= 1) {
            for (int i = 0; i < count; i++) {
                body.accept(i);
            }
            return;
        }
        POOL.submit(() -> IntStream.range(0, count).parallel().forEach(body)).join();
    }

    private static int clamp(int value, int min, int max) {
        return Math.min(Math.max(value, min), max);
    }

    static Tensor conv(Tensor input, Weights filter, float[] bias, int sy, int sx, int[] pad, int groups) {
        int channels = filter.shape()[0], perGroup = filter.shape()[1];
        int kh = filter.shape()[2], kw = filter.shape()[3];
        int height = (input.height() + pad[0] + pad[2] - kh) / sy + 1;
        int width = (input.width() + pad[1] + pad[3] - kw) / sx + 1;
        Tensor output = new Tensor(channels, height, width, new float[channels * height * width]);
        boolean depthwise = groups == input.channels() && channels == groups && groups > 1;
        if (depthwise) {
            forChannels(channels, c -> depthwise(input, output, filter.data(), bias, c, kh, kw, sy, sx, pad));
            return output;
        }

        int terms = perGroup * kh * kw;
        boolean pointwise = kh == 1 && kw == 1 && sy == 1 && sx == 1
                && pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0;
        float[] columns = pointwise ? input.data() : new float[input.channels() * kh * kw * output.plane()];
        if (!pointwise) {
            fillImageColumns(input, output, kh, kw, sy, sx, pad, columns);
        }
        int outPlane = output.plane();
        int channelsPerGroup = channels / groups;
        forChannels(channels, c -> {
            int rowStart = c * outPlane;
            Arrays.fill(output.data(), rowStart, rowStart + outPlane, bias == null ? 0f : bias[c]);
            int inputStart = c / channelsPerGroup * terms * outPlane;
            for (int k = 0; k < terms; k++) {
                float scale = filter.data()[c * terms + k];
                accumulate(output.data(), rowStart, columns, inputStart + k * outPlane, outPlane, scale);
            }
        });
        return output;
    }

    private static void fillImageColumns(Tensor input, Tensor output, int kh, int kw, int sy, int sx,
            int[] pad, float[] columns) {
        int outPlane = output.plane();
        int outWidth = output.width();
        int outHeight = output.height();
        forChannels(input.channels(), c -> {
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    int offset = (c * kh * kw + ky * kw + kx) * outPlane;
                    int dx = kx - pad[1];
                    int first = clamp((-dx + sx - 1) / sx, 0, outWidth);
                    int end = clamp((input.width() - dx + sx - 1) / sx, first, outWidth);
                    for (int y = 0; y < outHeight; y++) {
                        int rowStart = offset + y * outWidth;
                        int iy = y * sy + ky - pad[0];
                        if (iy < 0 || iy >= input.height()) {
                            Arrays.fill(columns, rowStart, rowStart + outWidth, 0f);
                            continue;
                        }
                        int source = c * input.plane() + iy * input.width() + dx;
                        Arrays.fill(columns, rowStart, rowStart + first, 0f);
                        int destination = rowStart + first;
                        for (int x = first; x < end; x++) {
                            columns[destination++] = input.data()[source + x * sx];
                        }
                        Arrays.fill(columns, rowStart + end, rowStart + outWidth, 0f);
                    }
                }
            }
        });
    }

    private static void depthwise(Tensor input, Tensor output, float[] filter, float[] bias, int channel,
            int kh, int kw, int sy, int sx, int[] pad) {
        int outPlane = output.plane();
        int rowStart = channel * outPlane;
        Arrays.fill(output.data(), rowStart, rowStart + outPlane, bias == null ? 0f : bias[channel]);
        int inputOffset = channel * input.plane();
        int outWidth = output.width();
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
                float scale = filter[(channel * kh + ky) * kw + kx];
                int dx = kx - pad[1];
                int first = Math.max(0, (-dx + sx - 1) / sx);
                int end = clamp((input.width() - dx + sx - 1) / sx, first, outWidth);
                for (int y = 0; y < output.height(); y++) {
                    int iy = y * sy + ky - pad[0];
                    if (iy < 0 || iy >= input.height() || first == end) {
                        continue;
                    }
                    int source = inputOffset + iy * input.width();
                    int rowBase = rowStart + y * outWidth;
                    if (sx == 1) {
                        accumulate(output.data(), rowBase + first, input.data(), source + first + dx,
                                end - first, scale);
                    } else {
                        for (int x = first; x < end; x++) {
                            output.data()[rowBase + x] += input.data()[source + x * sx + dx] * scale;
                        }
                    }
                }
            }
        }
    }

    private static void accumulate(float[] destination, int destinationOffset,
            float[] source, int sourceOffset, int length, float scale) {
        for (int i = 0; i < length; i++) {
            destination[destinationOffset + i] += source[sourceOffset + i] * scale;
        }
    }

    static void normalizeAndActivate(Tensor x, float[] scale, float[] offset, float[] slope) {
        for (int c = 0; c < x.channels(); c++) {
            int start = c * x.plane();
            int end = start + x.plane();
            float factor = scale[c], bias = offset[c], negativeScale = slope[c];
            for (int i = start; i < end; i++) {
                float value = x.data()[i] * factor + bias;
                x.data()[i] = value >= 0 ? value : value * negativeScale;
            }
        }
    }

    static Tensor batchNorm(Tensor x, float[] gamma, float[] beta, float[] mean, float[] variance,
            float epsilon) {
        float[] result = new float[x.data().length];
        for (int c = 0; c < x.channels(); c++) {
            float scale = gamma[c] / (float) Math.sqrt(variance[c] + epsilon);
            float offset = beta[c] - mean[c] * scale;
            int end = (c + 1) * x.plane();
            for (int i = c * x.plane(); i < end; i++) {
                result[i] = x.data()[i] * scale + offset;
            }
        }
        return x.withData(result);
    }

    static void reluInPlace(Tensor x) {
        float[] values = x.data();
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.max(0f, values[i]);
        }
    }

    static Tensor prelu(Tensor x, float[] slope) {
        float[] result = new float[x.data().length];
        for (int c = 0; c < x.channels(); c++) {
            float negativeScale = slope == null ? 0f : slope[c];
            int end = (c + 1) * x.plane();
            for (int i = c * x.plane(); i < end; i++) {
                float value = x.data()[i];
                result[i] = value >= 0 ? value : value * negativeScale;
            }
        }
        return x.withData(result);
    }

    static Tensor pool2(Tensor x) {
        int outHeight = x.height() / 2, outWidth = x.width() / 2;
        Tensor output = new Tensor(x.channels(), outHeight, outWidth,
                new float[x.channels() * outHeight * outWidth]);
        for (int c = 0; c < x.channels(); c++) {
            for (int y = 0; y < outHeight; y++) {
                for (int col = 0; col < outWidth; col++) {
                    int i = c * x.plane() + 2 * y * x.width() + 2 * col;
                    output.data()[c * output.plane() + y * outWidth + col] = Math.max(
                            Math.max(x.data()[i], x.data()[i + 1]),
                            Math.max(x.data()[i + x.width()], x.data()[i + x.width() + 1]));
                }
            }
        }
        return output;
    }

    static Tensor upsample2(Tensor x) {
        int outHeight = x.height() * 2, outWidth = x.width() * 2;
        Tensor output = new Tensor(x.channels(), outHeight, outWidth, new float[x.data().length * 4]);
        for (int c = 0; c < x.channels(); c++) {
            for (int y = 0; y < outHeight; y++) {
                for (int col = 0; col < outWidth; col++) {
                    output.data()[c * output.plane() + y * outWidth + col] =
                            x.data()[c * x.plane() + (y / 2) * x.width() + col / 2];
                }
            }
        }
        return output;
    }

    static Tensor toInterleaved(Tensor x) {
        float[] output = new float[x.data().length];
        for (int c = 0; c < x.channels(); c++) {
            for (int i = 0; i < x.plane(); i++) {
                output[i * x.channels() + c] = x.data()[c * x.plane() + i];
            }
        }
        return x.withData(output);
    }

    interface FloatOp {
        float apply(float value);
    }

    static Tensor map(Tensor x, FloatOp map) {
        float[] result = new float[x.data().length];
        for (int i = 0; i < result.length; i++) {
            result[i] = map.apply(x.data()[i]);
        }
        return x.withData(result);
    }

    static Tensor add(Tensor x, Tensor y) {
        float[] output = new float[x.data().length];
        for (int i = 0; i < output.length; i++) {
            output[i] = x.data()[i] + y.data()[i];
        }
        return x.withData(output);
    }

    static Tensor dense(Tensor x, Weights matrix, float[] bias) {
        float[] output = new float[matrix.shape()[0]];
        float[] input = x.data();
        float[] weights = matrix.data();
        forChannels(output.length, c -> {
            float sum = 0f;
            int start = c * input.length;
            for (int i = 0; i < input.length; i++) {
                sum += input[i] * weights[start + i];
            }
            output[c] = (bias == null ? 0f : bias[c]) + sum;
        });
        return new Tensor(output.length, 1, 1, output);
    }

    static float dot(float[] x, float[] y) {
        float sum = 0f;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }
}
