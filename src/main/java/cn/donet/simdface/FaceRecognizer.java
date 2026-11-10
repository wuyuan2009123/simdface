package cn.donet.simdface;

import java.io.IOException;
import java.util.Arrays;

/** SFace aligner and 128-dimensional embedding extractor. */
public final class FaceRecognizer {

    public static final String MODEL_FILE_NAME = "face_recognition_sface_2021dec.onnx";
    static final String MODEL_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79";
    public static final float REFERENCE_THRESHOLD = 0.363f;
    public static final int EMBEDDING_SIZE = 128;
    public static final String MODEL_ID = "sface-2021dec";
    public static final String VERIFICATION_PIPELINE_ID = "align5-6crop-v1";
    private static final int AlignedSize = 112;
    private static final double[] VerificationCropScales = {1.0, 0.93, 1.07};

    // SFace's five-point reference geometry, in the same order as YuNet.
    private static final FacePoint[] Template = {
            new FacePoint(38.2946f, 51.6963f), new FacePoint(73.5318f, 51.5014f),
            new FacePoint(56.0252f, 71.7366f), new FacePoint(41.5493f, 92.3655f),
            new FacePoint(70.7299f, 92.2041f)
    };

    private final ModelGraph model;

    public FaceRecognizer(String modelPath) throws IOException {
        this(modelPath, null);
    }

    public FaceRecognizer(String modelPath, VulkanInference vulkan) throws IOException {
        this(ModelGraph.readFile(modelPath), modelPath, vulkan);
    }

    /** Loads the pinned ONNX model from memory. */
    public FaceRecognizer(byte[] modelBytes) {
        this(modelBytes, MODEL_FILE_NAME, null);
    }

    public FaceRecognizer(byte[] modelBytes, VulkanInference vulkan) {
        this(modelBytes, MODEL_FILE_NAME, vulkan);
    }

    private FaceRecognizer(byte[] modelBytes, String label, VulkanInference vulkan) {
        model = new ModelGraph(modelBytes, label, MODEL_SHA256, "data", vulkan, "fc1");
    }

    /** Aligns the face and returns an L2-normalized, 128-dimensional SFace embedding. */
    public float[] embed(byte[] rgb, int width, int height, FaceLandmarks landmarks) {
        return embedCore(rgb, width, height, landmarks, false);
    }

    /** Combines three aligned crops and their mirrors into one normalized SFace embedding. */
    public float[] embedForVerification(byte[] rgb, int width, int height, FaceLandmarks landmarks) {
        return embedCore(rgb, width, height, landmarks, true);
    }

    private float[] embedCore(byte[] rgb, int width, int height, FaceLandmarks landmarks, boolean enhanced) {
        ImageSampling.validate(rgb, width, height);
        FacePoint[] points = landmarks.asArray();

        double sx = 0, sy = 0, tx = 0, ty = 0;
        for (int i = 0; i < points.length; i++) {
            sx += points[i].x();
            sy += points[i].y();
            tx += Template[i].x();
            ty += Template[i].y();
        }
        sx /= points.length;
        sy /= points.length;
        tx /= points.length;
        ty /= points.length;

        // Fit a rotation, uniform scale and translation from source to template.
        double dot = 0, cross = 0, norm = 0;
        for (int i = 0; i < points.length; i++) {
            double x = points[i].x() - sx, y = points[i].y() - sy;
            double u = Template[i].x() - tx, v = Template[i].y() - ty;
            dot += x * u + y * v;
            cross += x * v - y * u;
            norm += x * x + y * y;
        }
        double a = dot / norm, b = cross / norm;
        double determinant = a * a + b * b;
        if (!Double.isFinite(determinant) || determinant < 1e-12) {
            throw new IllegalArgumentException("Landmarks do not define a valid face transform.");
        }

        Tensor crop = new Tensor(3, AlignedSize, AlignedSize, new float[3 * AlignedSize * AlignedSize]);
        if (!enhanced) {
            sampleCrop(rgb, width, height, crop, 1, sx, sy, tx, ty, a, b, determinant);
            return normalize(model.run(crop)[0]);
        }

        Tensor mirror = new Tensor(3, AlignedSize, AlignedSize, new float[crop.data().length]);
        float[] combined = new float[EMBEDDING_SIZE];
        if (model.usesVulkan()) {
            // Submit all six crops in one command buffer.
            Tensor[] batch = new Tensor[VerificationCropScales.length * 2];
            int index = 0;
            for (double scale : VerificationCropScales) {
                Arrays.fill(crop.data(), 0f);
                sampleCrop(rgb, width, height, crop, scale, sx, sy, tx, ty, a, b, determinant);
                batch[index++] = new Tensor(crop.channels(), crop.height(), crop.width(),
                        crop.data().clone());
                mirrorCrop(crop, mirror);
                batch[index++] = new Tensor(mirror.channels(), mirror.height(), mirror.width(),
                        mirror.data().clone());
            }
            for (float[] embedding : model.runVulkanBatch(batch)) {
                addEmbedding(combined, embedding);
            }
            return normalize(combined);
        }

        for (double scale : VerificationCropScales) {
            // Out-of-frame pixels sample as zero, including when a wider crop follows an earlier crop.
            Arrays.fill(crop.data(), 0f);
            sampleCrop(rgb, width, height, crop, scale, sx, sy, tx, ty, a, b, determinant);
            addEmbedding(combined, model.run(crop)[0]);
            mirrorCrop(crop, mirror);
            addEmbedding(combined, model.run(mirror)[0]);
        }
        return normalize(combined);
    }

    private static void mirrorCrop(Tensor crop, Tensor mirror) {
        for (int row = 0; row < 3 * AlignedSize; row++) {
            for (int x = 0; x < AlignedSize; x++) {
                mirror.data()[row * AlignedSize + x] =
                        crop.data()[row * AlignedSize + AlignedSize - 1 - x];
            }
        }
    }

    private static void sampleCrop(byte[] pixels, int width, int height, Tensor crop, double scale,
            double sx, double sy, double tx, double ty, double a, double b, double determinant) {
        double center = (AlignedSize - 1) / 2.0;
        for (int y = 0; y < AlignedSize; y++) {
            for (int x = 0; x < AlignedSize; x++) {
                double u = (x - center) / scale + center - tx;
                double v = (y - center) / scale + center - ty;
                float sourceX = (float) (sx + (a * u + b * v) / determinant);
                float sourceY = (float) (sy + (a * v - b * u) / determinant);
                ImageSampling.writePixel(pixels, width, height, sourceX, sourceY,
                        crop, y * AlignedSize + x, false);
            }
        }
    }

    private static void addEmbedding(float[] combined, float[] embedding) {
        for (int i = 0; i < combined.length; i++) {
            combined[i] += embedding[i];
        }
    }

    private static float[] normalize(float[] embedding) {
        float length = (float) Math.sqrt(Kernels.dot(embedding, embedding));
        if (!Float.isFinite(length) || length <= 0) {
            throw new InvalidDataException("The model returned an invalid embedding.");
        }
        for (int i = 0; i < embedding.length; i++) {
            embedding[i] /= length;
        }
        return embedding;
    }

    public static float cosineSimilarity(float[] first, float[] second) {
        if (first.length != EMBEDDING_SIZE || second.length != EMBEDDING_SIZE) {
            throw new IllegalArgumentException("SFace embeddings must contain " + EMBEDDING_SIZE + " values.");
        }
        float lengthA = (float) Math.sqrt(Kernels.dot(first, first));
        float lengthB = (float) Math.sqrt(Kernels.dot(second, second));
        if (!Float.isFinite(lengthA) || !Float.isFinite(lengthB) || lengthA <= 0 || lengthB <= 0) {
            throw new IllegalArgumentException("Embeddings must be finite and nonzero.");
        }
        return Math.max(-1f, Math.min(1f, Kernels.dot(first, second) / lengthA / lengthB));
    }
}
