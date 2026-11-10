package cn.donet.simdface;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Reads only the ONNX fields and operators used by the two pinned models and executes the
 * graph with the CPU kernels.
 */
final class ModelGraph {

    private static final int MAX_MODEL_BYTES = 40_000_000;

    private final Map<String, Weights> weights = new HashMap<>();
    private final List<Layer> layers = new ArrayList<>();
    private final Map<String, Integer> lastUse = new HashMap<>();
    private final Map<Integer, ConvActivation> convActivations = new HashMap<>();
    private final String inputName;
    private final String[] outputNames;
    private final VulkanInference vulkan;

    static byte[] readFile(String path) throws IOException {
        Path file = Path.of(path);
        if (Files.size(file) > MAX_MODEL_BYTES) {
            throw new InvalidDataException("Unexpected model size: " + path);
        }
        return Files.readAllBytes(file);
    }

    ModelGraph(byte[] bytes, String label, String sha256, String inputName, String... outputNames) {
        this(bytes, label, sha256, inputName, null, outputNames);
    }

    ModelGraph(byte[] bytes, String label, String sha256, String inputName, VulkanInference vulkan,
            String... outputNames) {
        if (bytes.length > MAX_MODEL_BYTES) {
            throw new InvalidDataException("Unexpected model size: " + label);
        }
        if (!sha256Hex(bytes).equals(sha256.toLowerCase(Locale.ROOT))) {
            throw new InvalidDataException(
                    "Model SHA-256 mismatch: " + label + ". Use the version listed in the README.");
        }
        this.inputName = inputName;
        this.outputNames = outputNames;
        this.vulkan = vulkan;

        ProtoReader model = new ProtoReader(bytes);
        while (model.next()) {
            if (model.field != 7) {
                continue;
            }
            ProtoReader graph = model.slice();
            while (graph.next()) {
                if (graph.field == 1) {
                    layers.add(readLayer(graph.slice()));
                } else if (graph.field == 5) {
                    readWeights(graph.slice());
                }
            }
            break;
        }

        for (int i = 0; i < layers.size(); i++) {
            for (String name : layers.get(i).inputs) {
                if (!weights.containsKey(name)) {
                    lastUse.put(name, i);
                }
            }
        }
        for (String name : outputNames) {
            lastUse.put(name, layers.size());
        }

        if (vulkan == null) {
            prepareConvActivations();
        }
    }

    private static String sha256Hex(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable.", error);
        }
    }

    private void prepareConvActivations() {
        for (int i = 0; i + 1 < layers.size(); i++) {
            Layer conv = layers.get(i);
            Layer norm = layers.get(i + 1);
            // Only mutate a fresh convolution output with a single remaining consumer.
            if (!conv.op.equals("Conv") || !norm.inputs.get(0).equals(conv.output)
                    || lookupLastUse(conv.output) != i + 1) {
                continue;
            }
            if (norm.op.equals("Relu")) {
                convActivations.put(i, null);
                continue;
            }
            if (!norm.op.equals("BatchNormalization") || i + 2 == layers.size()) {
                continue;
            }
            Layer activation = layers.get(i + 2);
            if (!activation.op.equals("PRelu") || !activation.inputs.get(0).equals(norm.output)
                    || lookupLastUse(norm.output) != i + 2) {
                continue;
            }

            float[] gamma = weights.get(norm.inputs.get(1)).data();
            float[] beta = weights.get(norm.inputs.get(2)).data();
            float[] mean = weights.get(norm.inputs.get(3)).data();
            float[] variance = weights.get(norm.inputs.get(4)).data();
            float[] scale = new float[gamma.length];
            float[] offset = new float[gamma.length];
            for (int c = 0; c < scale.length; c++) {
                scale[c] = gamma[c] / (float) Math.sqrt(variance[c] + norm.epsilon);
                offset[c] = beta[c] - mean[c] * scale[c];
            }
            convActivations.put(i, new ConvActivation(scale, offset, weights.get(activation.inputs.get(1)).data()));
        }
    }

    private int lookupLastUse(String name) {
        Integer last = lastUse.get(name);
        return last == null ? -1 : last;
    }

    boolean usesVulkan() {
        return vulkan != null;
    }

    float[][] run(Tensor input) {
        if (vulkan != null) {
            return vulkan.run(this, input);
        }
        return runCpu(input);
    }

    float[][] runVulkanBatch(Tensor[] inputs) {
        return vulkan.runBatch(this, inputs);
    }

    private float[][] runCpu(Tensor input) {
        Map<String, Tensor> values = new HashMap<>();
        values.put(inputName, input);
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            List<String> args = layer.inputs;
            Tensor x = values.get(args.get(0));
            Tensor output = switch (layer.op) {
                case "Conv" -> Kernels.conv(x, weights.get(args.get(1)),
                        args.size() == 3 && !args.get(2).isEmpty() ? weights.get(args.get(2)).data() : null,
                        layer.strides[0], layer.strides[1], layer.pads, layer.group);
                case "BatchNormalization" -> Kernels.batchNorm(x, weights.get(args.get(1)).data(),
                        weights.get(args.get(2)).data(), weights.get(args.get(3)).data(),
                        weights.get(args.get(4)).data(), layer.epsilon);
                case "PRelu" -> Kernels.prelu(x, weights.get(args.get(1)).data());
                case "Relu" -> Kernels.prelu(x, null);
                case "MaxPool" -> Kernels.pool2(x);
                case "Resize" -> Kernels.upsample2(x);
                case "Transpose" -> Kernels.toInterleaved(x);
                case "Sigmoid" -> Kernels.map(x, value -> 1f / (1f + (float) Math.exp(-value)));
                case "Add" -> Kernels.add(x, values.get(args.get(1)));
                case "Sub" -> Kernels.map(x, value -> value - weights.get(args.get(1)).data()[0]);
                case "Mul" -> Kernels.map(x, value -> value * weights.get(args.get(1)).data()[0]);
                case "Gemm" -> Kernels.dense(x, weights.get(args.get(1)),
                        args.size() > 2 && !args.get(2).isEmpty() ? weights.get(args.get(2)).data() : null);
                // The following layers preserve the flat data in these graphs.
                case "Flatten", "Reshape", "Dropout" -> x;
                default -> throw new UnsupportedOperationException("Unsupported model operator: " + layer.op);
            };
            for (String name : args) {
                Integer last = lastUse.get(name);
                if (last != null && last == i) {
                    values.remove(name);
                }
            }

            if (convActivations.containsKey(i)) {
                ConvActivation activation = convActivations.get(i);
                if (activation == null) {
                    Kernels.reluInPlace(output);
                    i++;
                } else {
                    Kernels.normalizeAndActivate(output, activation.scale(), activation.offset(), activation.slope());
                    i += 2;
                }
            }
            values.put(layers.get(i).output, output);
        }
        float[][] result = new float[outputNames.length][];
        for (int i = 0; i < outputNames.length; i++) {
            result[i] = values.get(outputNames[i]).data();
        }
        return result;
    }

    /** Plans tensor offsets and compute dispatches for one input shape. */
    GraphPlan planVulkan(Tensor input) {
        int next = 0;
        Map<String, Integer> constants = new HashMap<>();
        List<float[]> uploads = new ArrayList<>();
        for (Map.Entry<String, Weights> entry : weights.entrySet()) {
            constants.put(entry.getKey(), next);
            uploads.add(entry.getValue().data());
            next = Math.addExact(next, entry.getValue().data().length);
        }

        int weightLength = next;
        TensorSlot inputSlot = new TensorSlot(next, input.channels(), input.height(), input.width());
        next = Math.addExact(next, inputSlot.length());
        Map<String, TensorSlot> slots = new HashMap<>();
        slots.put(inputName, inputSlot);
        List<Dispatch> dispatches = new ArrayList<>(layers.size());

        for (Layer layer : layers) {
            List<String> args = layer.inputs;
            TensorSlot src = slots.get(args.get(0));
            if (layer.op.equals("Flatten") || layer.op.equals("Reshape") || layer.op.equals("Dropout")) {
                slots.put(layer.output, src);
                continue;
            }

            int channels = src.channels(), height = src.height(), width = src.width();
            Dispatch p = new Dispatch();
            p.source = src.offset();
            p.destination = next;
            p.inChannels = src.channels();
            p.inHeight = src.height();
            p.inWidth = src.width();

            switch (layer.op) {
                case "Conv" -> {
                    int[] shape = weights.get(args.get(1)).shape();
                    channels = shape[0];
                    p.kernelHeight = shape[2];
                    p.kernelWidth = shape[3];
                    p.strideY = layer.strides[0];
                    p.strideX = layer.strides[1];
                    p.padY = layer.pads[0];
                    p.padX = layer.pads[1];
                    p.groups = layer.group;
                    height = (height + p.padY + layer.pads[2] - p.kernelHeight) / p.strideY + 1;
                    width = (width + p.padX + layer.pads[3] - p.kernelWidth) / p.strideX + 1;
                    p.weights = constants.get(args.get(1));
                    p.bias = args.size() == 3 ? constants.getOrDefault(args.get(2), -1) : -1;
                    p.op = p.kernelHeight == 1 && p.kernelWidth == 1 && p.groups == 1
                            && p.strideY == 1 && p.strideX == 1
                            && layer.pads[0] == 0 && layer.pads[1] == 0
                            && layer.pads[2] == 0 && layer.pads[3] == 0 ? 13 : 0;
                }
                case "BatchNormalization" -> {
                    p.op = 1;
                    p.weights = constants.get(args.get(1));
                    p.bias = constants.get(args.get(2));
                    p.auxiliary = constants.get(args.get(3));
                    p.extra = constants.get(args.get(4));
                    p.scalar = layer.epsilon;
                }
                case "PRelu" -> {
                    p.op = 2;
                    p.weights = constants.get(args.get(1));
                }
                case "Relu" -> p.op = 3;
                case "MaxPool" -> {
                    p.op = 4;
                    height /= 2;
                    width /= 2;
                }
                case "Resize" -> {
                    p.op = 5;
                    height *= 2;
                    width *= 2;
                }
                case "Transpose" -> p.op = 6;
                case "Sigmoid" -> p.op = 7;
                case "Add" -> {
                    p.op = 8;
                    p.auxiliary = slots.get(args.get(1)).offset();
                }
                case "Sub" -> {
                    p.op = 9;
                    p.scalar = weights.get(args.get(1)).data()[0];
                }
                case "Mul" -> {
                    p.op = 10;
                    p.scalar = weights.get(args.get(1)).data()[0];
                }
                case "Gemm" -> {
                    p.op = 12;
                    p.weights = constants.get(args.get(1));
                    p.bias = constants.getOrDefault(args.get(2), -1);
                    channels = weights.get(args.get(1)).shape()[0];
                    height = 1;
                    width = 1;
                }
                default -> throw new UnsupportedOperationException(
                        "Unsupported Vulkan model operator: " + layer.op);
            }

            p.outChannels = channels;
            p.outHeight = height;
            p.outWidth = width;
            TensorSlot output = new TensorSlot(next, channels, height, width);
            slots.put(layer.output, output);
            next = Math.addExact(next, output.length());
            dispatches.add(p);
        }

        TensorSlot[] outputs = new TensorSlot[outputNames.length];
        for (int i = 0; i < outputNames.length; i++) {
            outputs[i] = slots.get(outputNames[i]);
        }
        return new GraphPlan(inputSlot, outputs, dispatches.toArray(Dispatch[]::new),
                uploads.toArray(float[][]::new), weightLength, next);
    }

    private void readWeights(ProtoReader reader) {
        List<Integer> shape = new ArrayList<>(4);
        List<Float> scalarValues = new ArrayList<>();
        String name = "";
        int type = 0;
        int rawOffset = -1;
        int rawLength = 0;
        while (reader.next()) {
            switch (reader.field) {
                case 1 -> readInts(shape, reader);
                case 2 -> type = (int) reader.number;
                case 4 -> {
                    if (reader.valueLength == 4) {
                        scalarValues.add(reader.floatLE());
                    } else {
                        rawOffset = reader.valueOffset;
                        rawLength = reader.valueLength;
                    }
                }
                case 8 -> name = reader.utf8();
                case 9 -> {
                    rawOffset = reader.valueOffset;
                    rawLength = reader.valueLength;
                }
                default -> {
                }
            }
        }
        if (type != 1) {
            // Non-float initializers are not used by the pinned models.
            return;
        }
        float[] data;
        if (rawOffset >= 0) {
            if ((rawLength & 3) != 0) {
                throw new InvalidDataException("Invalid float initializer: " + name);
            }
            data = new float[rawLength / 4];
            for (int i = 0; i < data.length; i++) {
                data[i] = Float.intBitsToFloat(reader.intLE(rawOffset + i * 4));
            }
        } else {
            data = new float[scalarValues.size()];
            for (int i = 0; i < data.length; i++) {
                data[i] = scalarValues.get(i);
            }
        }
        int[] dims = new int[shape.size()];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = shape.get(i);
        }
        weights.put(name, new Weights(dims, data));
    }

    private static Layer readLayer(ProtoReader reader) {
        Layer layer = new Layer();
        while (reader.next()) {
            switch (reader.field) {
                case 1 -> layer.inputs.add(reader.utf8());
                case 2 -> layer.output = reader.utf8();
                case 4 -> layer.op = reader.utf8();
                case 5 -> readAttribute(layer, reader.slice());
                default -> {
                }
            }
        }
        return layer;
    }

    private static void readAttribute(Layer layer, ProtoReader reader) {
        String name = "";
        int integer = 0;
        float scalar = 0;
        List<Integer> integers = new ArrayList<>();
        while (reader.next()) {
            switch (reader.field) {
                case 1 -> name = reader.utf8();
                case 2 -> scalar = reader.floatLE();
                case 3 -> integer = (int) reader.number;
                case 8 -> readInts(integers, reader);
                default -> {
                }
            }
        }
        switch (name) {
            case "group" -> layer.group = integer;
            case "strides" -> {
                layer.strides = new int[] {
                        integers.size() > 0 ? integers.get(0) : 1,
                        integers.size() > 1 ? integers.get(1) : 1
                };
            }
            case "pads" -> {
                int[] pads = new int[4];
                for (int i = 0; i < Math.min(4, integers.size()); i++) {
                    pads[i] = integers.get(i);
                }
                layer.pads = pads;
            }
            case "epsilon" -> layer.epsilon = scalar;
            default -> {
            }
        }
    }

    private static void readInts(List<Integer> result, ProtoReader reader) {
        if (reader.valueLength == 0) {
            result.add((int) reader.number);
            return;
        }
        ProtoReader packed = reader.slice();
        while (!packed.atEnd()) {
            result.add((int) packed.readVarint());
        }
    }

    private record ConvActivation(float[] scale, float[] offset, float[] slope) {
    }

    private static final class Layer {
        String op = "";
        String output = "";
        final List<String> inputs = new ArrayList<>();
        int group = 1;
        int[] strides = {1, 1};
        int[] pads = {0, 0, 0, 0};
        float epsilon = 1e-5f;
    }
}
