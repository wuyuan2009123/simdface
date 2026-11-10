package cn.donet.simdface;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** YuNet face detector and five-point locator. */
public final class FaceDetector {

    public static final String MODEL_FILE_NAME = "face_detection_yunet_2023mar.onnx";
    static final String MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4";
    public static final float DEFAULT_SCORE_THRESHOLD = 0.8f;
    private static final int[] Strides = {8, 16, 32};

    private final ModelGraph model;

    public FaceDetector(String modelPath) throws IOException {
        this(modelPath, null);
    }

    public FaceDetector(String modelPath, VulkanInference vulkan) throws IOException {
        this(ModelGraph.readFile(modelPath), modelPath, vulkan);
    }

    /** Loads the pinned ONNX model from memory. */
    public FaceDetector(byte[] modelBytes) {
        this(modelBytes, MODEL_FILE_NAME, null);
    }

    public FaceDetector(byte[] modelBytes, VulkanInference vulkan) {
        this(modelBytes, MODEL_FILE_NAME, vulkan);
    }

    private FaceDetector(byte[] modelBytes, String label, VulkanInference vulkan) {
        model = new ModelGraph(modelBytes, label, MODEL_SHA256, "input", vulkan,
                "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
                "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32");
    }

    /** Detects faces in tightly packed RGB24 data, highest confidence first. */
    public List<DetectedFace> detect(byte[] rgb, int width, int height) {
        return detect(rgb, width, height, DEFAULT_SCORE_THRESHOLD);
    }

    public List<DetectedFace> detect(byte[] rgb, int width, int height, float scoreThreshold) {
        ImageSampling.validate(rgb, width, height);
        if (!(scoreThreshold > 0 && scoreThreshold <= 1)) {
            throw new IllegalArgumentException("scoreThreshold must be in (0, 1].");
        }

        double scale = Math.min(1, 640.0 / Math.max(width, height));
        int resizedWidth = Math.max(1, (int) Math.rint(width * scale));
        int resizedHeight = Math.max(1, (int) Math.rint(height * scale));
        int paddedWidth = (resizedWidth + 31) / 32 * 32;
        int paddedHeight = (resizedHeight + 31) / 32 * 32;
        Tensor input = new Tensor(3, paddedHeight, paddedWidth, new float[3 * paddedWidth * paddedHeight]);
        float scaleX = (float) width / resizedWidth, scaleY = (float) height / resizedHeight;
        for (int y = 0; y < resizedHeight; y++) {
            for (int x = 0; x < resizedWidth; x++) {
                int index = y * paddedWidth + x;
                if (scale == 1) {
                    int source = (y * width + x) * 3;
                    input.data()[index] = rgb[source + 2] & 0xFF;
                    input.data()[input.plane() + index] = rgb[source + 1] & 0xFF;
                    input.data()[2 * input.plane() + index] = rgb[source] & 0xFF;
                } else {
                    float sourceX = Math.max(0f, Math.min(width - 1f, (x + 0.5f) * scaleX - 0.5f));
                    float sourceY = Math.max(0f, Math.min(height - 1f, (y + 0.5f) * scaleY - 0.5f));
                    ImageSampling.writePixel(rgb, width, height, sourceX, sourceY, input, index, true);
                }
            }
        }

        float[][] predictions = model.run(input);
        List<DetectedFace> candidates = new ArrayList<>();
        for (int level = 0; level < Strides.length; level++) {
            int stride = Strides[level];
            int columns = paddedWidth / stride;
            float[] cls = predictions[level], obj = predictions[level + 3];
            float[] box = predictions[level + 6], points = predictions[level + 9];
            for (int i = 0; i < cls.length; i++) {
                float score = (float) Math.sqrt(cls[i] * obj[i]);
                if (!(score >= scoreThreshold)) {
                    continue;
                }
                int row = i / columns, col = i % columns;
                float cx = (col + box[i * 4]) * stride;
                float cy = (row + box[i * 4 + 1]) * stride;
                if (cx < 0 || cx >= resizedWidth || cy < 0 || cy >= resizedHeight) {
                    continue;
                }
                float w = (float) Math.exp(box[i * 4 + 2]) * stride;
                float h = (float) Math.exp(box[i * 4 + 3]) * stride;
                if (!Float.isFinite(w) || !Float.isFinite(h) || w <= 0 || h <= 0) {
                    continue;
                }
                FaceLandmarks landmarks = new FaceLandmarks(
                        pointAt(points, i, 0, row, col, stride, scaleX, scaleY),
                        pointAt(points, i, 2, row, col, stride, scaleX, scaleY),
                        pointAt(points, i, 4, row, col, stride, scaleX, scaleY),
                        pointAt(points, i, 6, row, col, stride, scaleX, scaleY),
                        pointAt(points, i, 8, row, col, stride, scaleX, scaleY));
                candidates.add(new DetectedFace((cx - w / 2) * scaleX, (cy - h / 2) * scaleY,
                        w * scaleX, h * scaleY, score, landmarks));
            }
        }

        Integer[] order = new Integer[candidates.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (left, right) -> {
            int scoreOrder = Float.compare(candidates.get(right).score(), candidates.get(left).score());
            return scoreOrder == 0 ? Integer.compare(left, right) : scoreOrder;
        });

        List<DetectedFace> faces = new ArrayList<>();
        int count = Math.min(order.length, 5000);
        for (int i = 0; i < count; i++) {
            DetectedFace candidate = candidates.get(order[i]);
            boolean overlaps = false;
            for (DetectedFace face : faces) {
                overlaps = intersectionOverUnion(face, candidate) > 0.3f;
                if (overlaps) {
                    break;
                }
            }
            if (!overlaps) {
                faces.add(candidate);
            }
        }
        return List.copyOf(faces);
    }

    private static FacePoint pointAt(float[] points, int i, int offset, int row, int col,
            int stride, float scaleX, float scaleY) {
        return new FacePoint((col + points[i * 10 + offset]) * stride * scaleX,
                (row + points[i * 10 + offset + 1]) * stride * scaleY);
    }

    private static float intersectionOverUnion(DetectedFace a, DetectedFace b) {
        float w = Math.max(0, Math.min(a.x() + a.width(), b.x() + b.width()) - Math.max(a.x(), b.x()));
        float h = Math.max(0, Math.min(a.y() + a.height(), b.y() + b.height()) - Math.max(a.y(), b.y()));
        float overlap = w * h;
        return overlap / (a.width() * a.height() + b.width() * b.height() - overlap);
    }
}
