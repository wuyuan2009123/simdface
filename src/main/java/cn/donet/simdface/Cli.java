package cn.donet.simdface;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/** detect / compare command line interface. */
public final class Cli {

    private static final String Usage = """
            SimdFace 0.1.0
            simdface detect <image> [--models <directory>] [--score <0..1>] [--backend cpu|vulkan]
            simdface compare <first> <second> [--models <directory>] [--score <0..1>] [--threshold <-1..1>] [--backend cpu|vulkan]
            """;

    private Cli() {
    }

    public static void main(String[] args) {
        System.exit(run(args));
    }

    static int run(String[] args) {
        if (args.length == 0 || (args.length == 1 && (args[0].equals("--help") || args[0].equals("-h")))) {
            System.out.print(Usage);
            return 0;
        }
        try {
            boolean compare = args[0].equals("compare");
            int imageCount = compare ? 2 : 1;
            if ((!compare && !args[0].equals("detect")) || args.length <= imageCount) {
                throw new IllegalArgumentException(Usage);
            }
            for (int i = 1; i <= imageCount; i++) {
                if (args[i].startsWith("--")) {
                    throw new IllegalArgumentException(Usage);
                }
            }

            String models = "models";
            String backend = "cpu";
            float score = FaceDetector.DEFAULT_SCORE_THRESHOLD;
            float threshold = FaceRecognizer.REFERENCE_THRESHOLD;
            for (int i = imageCount + 1; i < args.length; i += 2) {
                if (i + 1 == args.length) {
                    throw new IllegalArgumentException("Missing value for " + args[i] + ".");
                }
                String value = args[i + 1];
                switch (args[i]) {
                    case "--models" -> models = value;
                    case "--backend" -> {
                        if (!value.equals("cpu") && !value.equals("vulkan")) {
                            throw new IllegalArgumentException("--backend must be cpu or vulkan.");
                        }
                        backend = value;
                    }
                    case "--score" -> score = readNumber(value, "--score", 0, 1, true);
                    case "--threshold" -> {
                        if (!compare) {
                            throw new IllegalArgumentException("Unknown option: " + args[i]);
                        }
                        threshold = readNumber(value, "--threshold", -1, 1, false);
                    }
                    default -> throw new IllegalArgumentException("Unknown option: " + args[i]);
                }
            }

            VulkanInference vulkan = backend.equals("vulkan") ? new VulkanInference() : null;
            try {
                FaceDetector detector = vulkan == null
                        ? new FaceDetector(Path.of(models, FaceDetector.MODEL_FILE_NAME).toString())
                        : new FaceDetector(Path.of(models, FaceDetector.MODEL_FILE_NAME).toString(), vulkan);
                if (!compare) {
                    RgbImage image = ImageDecoder.readFile(args[1]);
                    List<DetectedFace> faces = detector.detect(
                            image.rgb(), image.width(), image.height(), score);
                    System.out.println(detectJson(image.width(), image.height(), faces));
                    return 0;
                }

                FaceRecognizer[] recognizer = new FaceRecognizer[1];
                float[] firstEmbedding = embedFile(args[1], detector, recognizer, models, score, vulkan);
                float[] secondEmbedding = embedFile(args[2], detector, recognizer, models, score, vulkan);
                float similarity = FaceRecognizer.cosineSimilarity(firstEmbedding, secondEmbedding);
                System.out.println("{\"similarity\":" + Json.number(similarity)
                        + ",\"threshold\":" + Json.number(threshold)
                        + ",\"isMatch\":" + (similarity >= threshold) + "}");
                return 0;
            } finally {
                if (vulkan != null) {
                    vulkan.close();
                }
            }
        } catch (IllegalArgumentException | IOException | InvalidDataException
                | UnsupportedOperationException | IllegalStateException error) {
            System.err.println(error.getMessage());
            return 2;
        }
    }

    private static float[] embedFile(String path, FaceDetector detector, FaceRecognizer[] holder,
            String models, float score, VulkanInference vulkan) throws IOException {
        RgbImage image = ImageDecoder.readFile(path);
        List<DetectedFace> faces = detector.detect(image.rgb(), image.width(), image.height(), score);
        if (faces.size() != 1) {
            throw new InvalidDataException(
                    "Expected exactly one face but found " + faces.size() + ": " + path);
        }
        if (holder[0] == null) {
            String modelPath = Path.of(models, FaceRecognizer.MODEL_FILE_NAME).toString();
            holder[0] = vulkan == null ? new FaceRecognizer(modelPath)
                    : new FaceRecognizer(modelPath, vulkan);
        }
        return holder[0].embed(image.rgb(), image.width(), image.height(), faces.get(0).landmarks());
    }

    private static float readNumber(String text, String option, float min, float max, boolean excludeMinimum) {
        float number;
        try {
            number = Float.parseFloat(text);
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException("Invalid value for " + option + ": " + text);
        }
        if (!Float.isFinite(number) || number < min || number > max
                || (excludeMinimum && number == min)) {
            throw new IllegalArgumentException("Invalid value for " + option + ": " + text);
        }
        return number;
    }

    static String detectJson(int width, int height, List<DetectedFace> faces) {
        StringBuilder json = new StringBuilder();
        json.append("{\"width\":").append(width)
                .append(",\"height\":").append(height)
                .append(",\"faces\":[");
        for (int i = 0; i < faces.size(); i++) {
            if (i > 0) {
                json.append(',');
            }
            json.append(faceJson(faces.get(i)));
        }
        return json.append("]}").toString();
    }

    static String faceJson(DetectedFace face) {
        FaceLandmarks marks = face.landmarks();
        return "{\"x\":" + Json.number(face.x())
                + ",\"y\":" + Json.number(face.y())
                + ",\"width\":" + Json.number(face.width())
                + ",\"height\":" + Json.number(face.height())
                + ",\"score\":" + Json.number(face.score())
                + ",\"landmarks\":{"
                + "\"rightEye\":" + pointJson(marks.rightEye())
                + ",\"leftEye\":" + pointJson(marks.leftEye())
                + ",\"nose\":" + pointJson(marks.nose())
                + ",\"rightMouth\":" + pointJson(marks.rightMouth())
                + ",\"leftMouth\":" + pointJson(marks.leftMouth())
                + "}}";
    }

    private static String pointJson(FacePoint point) {
        return "{\"x\":" + Json.number(point.x()) + ",\"y\":" + Json.number(point.y()) + "}";
    }
}
