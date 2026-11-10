package cn.donet.simdface.web;

import cn.donet.simdface.DetectedFace;
import cn.donet.simdface.FaceDetector;
import cn.donet.simdface.FaceRecognizer;
import cn.donet.simdface.ImageDecoder;
import cn.donet.simdface.InvalidDataException;
import cn.donet.simdface.Json;
import cn.donet.simdface.RgbImage;
import cn.donet.simdface.VulkanInference;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

/** HTTP test bench: static frontend plus the /api endpoints. */
public final class WebServer {

    private static final float DefaultSamePersonThreshold = 0.68f;
    private static final int MaxFileBytes = ImageDecoder.MaxFileBytes;
    private static final int MaxBodyBytes = 2 * MaxFileBytes + 64 * 1024;

    private final String modelDirectory;
    private final FaceDetector detector;
    private final FaceRecognizer recognizer;
    private final Semaphore inferenceSlot = new Semaphore(1);
    private volatile GpuModels gpu;

    private WebServer(String modelDirectory) throws IOException {
        this.modelDirectory = modelDirectory;
        detector = new FaceDetector(Path.of(modelDirectory, FaceDetector.MODEL_FILE_NAME).toString());
        recognizer = new FaceRecognizer(Path.of(modelDirectory, FaceRecognizer.MODEL_FILE_NAME).toString());
    }

    private synchronized GpuModels gpuModels() throws IOException {
        if (gpu == null) {
            VulkanInference device = new VulkanInference();
            try {
                gpu = new GpuModels(device,
                        new FaceDetector(Path.of(modelDirectory, FaceDetector.MODEL_FILE_NAME).toString(), device),
                        new FaceRecognizer(Path.of(modelDirectory, FaceRecognizer.MODEL_FILE_NAME).toString(),
                                device));
            } catch (IOException | RuntimeException error) {
                device.close();
                throw error;
            }
        }
        return gpu;
    }

    private synchronized void close() {
        if (gpu != null) {
            gpu.device().close();
            gpu = null;
        }
    }

    public static void main(String[] args) throws IOException {
        int port = 5088;
        String host = "127.0.0.1";
        String models = "models";
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 == args.length) {
                throw new IllegalArgumentException("Missing value for " + args[i] + ".");
            }
            switch (args[i]) {
                case "--port" -> port = Integer.parseInt(args[i + 1]);
                case "--host" -> host = args[i + 1];
                case "--models" -> models = args[i + 1];
                default -> throw new IllegalArgumentException("Unknown option: " + args[i]);
            }
        }

        WebServer app = new WebServer(models);
        Runtime.getRuntime().addShutdownHook(new Thread(app::close));
        HttpServer server = HttpServer.create(new InetSocketAddress(host, port), 0);
        server.createContext("/", app::handle);
        server.setExecutor(Executors.newFixedThreadPool(
                Math.max(4, Math.min(8, Runtime.getRuntime().availableProcessors()))));
        server.start();
        System.out.println("SimdFace Java listening on http://" + host + ":" + port + "/");
    }

    private void handle(HttpExchange exchange) throws IOException {
        try {
            String path = exchange.getRequestURI().getPath();
            if (path.startsWith("/api/")) {
                handleApi(exchange, path.substring("/api/".length()));
            } else {
                handleStatic(exchange, path);
            }
        } catch (InvalidDataException | IllegalArgumentException error) {
            sendJson(exchange, 400, errorJson(error.getMessage()));
        } catch (Throwable error) {
            sendJson(exchange, 500, errorJson("服务器内部错误：" + error.getMessage()));
        } finally {
            exchange.close();
        }
    }

    private void handleApi(HttpExchange exchange, String endpoint) throws IOException {
        if (!"POST".equals(exchange.getRequestMethod())) {
            sendJson(exchange, 405, errorJson("Method not allowed."));
            return;
        }
        switch (endpoint) {
            case "detect" -> analyze(exchange, PhotoMode.DETECT);
            case "compare" -> analyze(exchange, PhotoMode.COMPARE);
            case "extract" -> analyze(exchange, PhotoMode.EXTRACT);
            case "vectors" -> compareVectors(exchange);
            default -> sendJson(exchange, 404, errorJson("Not found."));
        }
    }

    private void analyze(HttpExchange exchange, PhotoMode mode) throws IOException {
        if (!"1".equals(exchange.getRequestHeaders().getFirst("X-SimdFace-Request"))) {
            sendJson(exchange, 400, errorJson("缺少请求标记。"));
            return;
        }
        Map<String, String> query = query(exchange.getRequestURI());
        String backend = query.getOrDefault("backend", "cpu");
        if (!backend.equals("cpu") && !backend.equals("vulkan")) {
            sendJson(exchange, 400, errorJson("请选择 CPU 或 Vulkan。"));
            return;
        }
        boolean useVulkan = backend.equals("vulkan");

        boolean compare = mode == PhotoMode.COMPARE;
        float score = readThreshold(query, "score", FaceDetector.DEFAULT_SCORE_THRESHOLD, 0);
        if (score == 0) {
            throw new InvalidDataException("检测阈值必须大于 0。");
        }
        float threshold = compare
                ? readThreshold(query, "threshold", DefaultSamePersonThreshold, -1) : 0;

        if (!inferenceSlot.tryAcquire()) {
            sendJson(exchange, 429, errorJson("正在处理另一张图片，请稍后再试。"));
            return;
        }
        try {
            FaceDetector activeDetector = detector;
            FaceRecognizer activeRecognizer = recognizer;
            String deviceName = null;
            if (useVulkan) {
                try {
                    GpuModels models = gpuModels();
                    activeDetector = models.detector();
                    activeRecognizer = models.recognizer();
                    deviceName = models.device().deviceName();
                } catch (IOException | UnsupportedOperationException | IllegalStateException error) {
                    sendJson(exchange, 503,
                            errorJson("Vulkan 执行失败，请尝试切换到 CPU。" + error.getMessage()));
                    return;
                }
            }

            int count = compare ? 2 : 1;
            List<Multipart.Part> files = readFiles(exchange, count, MaxFileBytes);
            ImageResult[] images = new ImageResult[count];
            float[][] embeddings = new float[count][];
            long decodeNanos = 0, inferenceNanos = 0;
            for (int i = 0; i < count; i++) {
                String label = i == 0 ? "照片 A" : "照片 B";
                long start = System.nanoTime();
                RgbImage image;
                try {
                    image = ImageDecoder.decode(files.get(i).data(), label);
                } catch (InvalidDataException error) {
                    throw new InvalidDataException("无法解码" + label + "。请选择有效图片。", error);
                }
                decodeNanos += System.nanoTime() - start;

                start = System.nanoTime();
                List<DetectedFace> faces = activeDetector.detect(
                        image.rgb(), image.width(), image.height(), score);
                images[i] = new ImageResult(image.width(), image.height(), faces);
                if (mode != PhotoMode.DETECT && faces.size() == 1 && (i == 0 || embeddings[0] != null)) {
                    embeddings[i] = activeRecognizer.embedForVerification(
                            image.rgb(), image.width(), image.height(), faces.get(0).landmarks());
                }
                inferenceNanos += System.nanoTime() - start;
            }

            String message = null;
            Feature feature = null;
            Comparison comparison = null;
            long start = System.nanoTime();
            if (compare) {
                if (embeddings[0] != null && embeddings[1] != null) {
                    comparison = compare(embeddings[0], embeddings[1], threshold);
                } else {
                    message = "比对要求每张图恰好一张脸。照片 A 检出 " + images[0].faces().size()
                            + " 张，照片 B 检出 " + images[1].faces().size() + " 张。";
                }
            } else if (mode == PhotoMode.EXTRACT) {
                if (embeddings[0] != null) {
                    feature = new Feature(FaceRecognizer.MODEL_ID,
                            FaceRecognizer.VERIFICATION_PIPELINE_ID, embeddings[0]);
                } else {
                    message = "提取特征要求恰好一张脸，检出了 " + images[0].faces().size() + " 张。";
                }
            }
            inferenceNanos += System.nanoTime() - start;

            String json = analysisJson(images, inferenceNanos / 1_000_000, decodeNanos / 1_000_000,
                    backend, deviceName, feature, comparison, message);
            sendJson(exchange, message == null ? 200 : 422, json);
        } catch (UnsupportedOperationException | IllegalStateException error) {
            if (useVulkan) {
                sendJson(exchange, 503,
                        errorJson("Vulkan 执行失败，请尝试切换到 CPU。" + error.getMessage()));
                return;
            }
            throw error;
        } finally {
            inferenceSlot.release();
        }
    }

    private void compareVectors(HttpExchange exchange) throws IOException {
        if (!"1".equals(exchange.getRequestHeaders().getFirst("X-SimdFace-Request"))) {
            sendJson(exchange, 400, errorJson("缺少请求标记。"));
            return;
        }
        Map<String, String> query = query(exchange.getRequestURI());
        float threshold = readThreshold(query, "threshold", DefaultSamePersonThreshold, -1);
        long start = System.nanoTime();
        List<Multipart.Part> files = readFiles(exchange, 2, 16 * 1024);
        float[] first = readFeature(files.get(0).data());
        float[] second = readFeature(files.get(1).data());
        Comparison comparison = compare(first, second, threshold);
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;
        sendJson(exchange, 200, "{\"backend\":\"cpu\",\"elapsedMs\":" + elapsedMs
                + ",\"comparison\":" + comparisonJson(comparison) + "}");
    }

    private static float[] readFeature(byte[] data) {
        Map<String, Object> feature;
        try {
            feature = Json.parseObject(new String(data, StandardCharsets.UTF_8));
        } catch (InvalidDataException error) {
            throw new InvalidDataException("特征 JSON 格式无效。", error);
        }
        Object values = feature.get("values");
        if (!FaceRecognizer.MODEL_ID.equals(feature.get("model"))
                || !FaceRecognizer.VERIFICATION_PIPELINE_ID.equals(feature.get("pipeline"))
                || !(values instanceof List<?> list)
                || list.size() != FaceRecognizer.EMBEDDING_SIZE) {
            throw new InvalidDataException("请选择由当前 SFace 提取流程导出的 128 维 JSON 特征。");
        }
        float[] result = new float[FaceRecognizer.EMBEDDING_SIZE];
        for (int i = 0; i < result.length; i++) {
            if (!(list.get(i) instanceof Number number)) {
                throw new InvalidDataException("请选择由当前 SFace 提取流程导出的 128 维 JSON 特征。");
            }
            result[i] = number.floatValue();
        }
        return result;
    }

    private static Comparison compare(float[] first, float[] second, float threshold) {
        float similarity = FaceRecognizer.cosineSimilarity(first, second);
        float differentThreshold = Math.min(FaceRecognizer.REFERENCE_THRESHOLD, threshold);
        boolean isMatch = similarity >= threshold;
        String decision = isMatch ? "same" : similarity < differentThreshold ? "different" : "uncertain";
        return new Comparison(similarity, threshold, differentThreshold, isMatch, decision);
    }

    private List<Multipart.Part> readFiles(HttpExchange exchange, int count, int fileLimit)
            throws IOException {
        String contentType = exchange.getRequestHeaders().getFirst("Content-Type");
        if (contentType == null
                || !contentType.toLowerCase(Locale.ROOT).startsWith("multipart/form-data")) {
            throw new InvalidDataException("请选择要上传的文件。");
        }
        String boundary = null;
        for (String token : contentType.split(";")) {
            String trimmed = token.trim();
            if (trimmed.regionMatches(true, 0, "boundary=", 0, "boundary=".length())) {
                boundary = trimmed.substring("boundary=".length());
                if (boundary.length() >= 2 && boundary.startsWith("\"") && boundary.endsWith("\"")) {
                    boundary = boundary.substring(1, boundary.length() - 1);
                }
            }
        }
        if (boundary == null || boundary.isEmpty() || boundary.length() > 128) {
            throw new InvalidDataException("文件上传格式无效。");
        }
        byte[] body = readBody(exchange.getRequestBody(), MaxBodyBytes);
        return Multipart.parse(body, boundary, count, fileLimit);
    }

    private static byte[] readBody(InputStream input, int limit) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[16 * 1024];
        int total = 0;
        int read;
        while ((read = input.read(buffer)) >= 0) {
            total += read;
            if (total > limit) {
                throw new InvalidDataException("请求体过大。");
            }
            output.write(buffer, 0, read);
        }
        return output.toByteArray();
    }

    private static float readThreshold(Map<String, String> query, String key,
            float defaultValue, float min) {
        String input = query.get(key);
        if (input == null) {
            return defaultValue;
        }
        float value;
        try {
            value = Float.parseFloat(input);
        } catch (NumberFormatException error) {
            throw new InvalidDataException("阈值超出范围，请检查输入。");
        }
        if (!Float.isFinite(value) || value < min || value > 1) {
            throw new InvalidDataException("阈值超出范围，请检查输入。");
        }
        return value;
    }

    private static Map<String, String> query(URI uri) {
        Map<String, String> result = new HashMap<>();
        String raw = uri.getRawQuery();
        if (raw == null || raw.isEmpty()) {
            return result;
        }
        for (String pair : raw.split("&")) {
            int equals = pair.indexOf('=');
            String key = equals < 0 ? pair : pair.substring(0, equals);
            String value = equals < 0 ? "" : pair.substring(equals + 1);
            result.put(URLDecoder.decode(key, StandardCharsets.UTF_8),
                    URLDecoder.decode(value, StandardCharsets.UTF_8));
        }
        return result;
    }

    private void handleStatic(HttpExchange exchange, String path) throws IOException {
        String method = exchange.getRequestMethod();
        if (!"GET".equals(method) && !"HEAD".equals(method)) {
            sendJson(exchange, 405, errorJson("Method not allowed."));
            return;
        }
        if (path.contains("..")) {
            sendJson(exchange, 404, errorJson("Not found."));
            return;
        }
        String name = path.equals("/") || path.isEmpty() ? "/index.html" : path;
        try (InputStream input = WebServer.class.getResourceAsStream("/wwwroot" + name)) {
            if (input == null) {
                sendJson(exchange, 404, errorJson("Not found."));
                return;
            }
            byte[] bytes = input.readAllBytes();
            Headers headers = exchange.getResponseHeaders();
            headers.set("Content-Type", contentType(name));
            headers.set("Cache-Control", "no-cache");
            exchange.sendResponseHeaders(200, bytes.length);
            if (!"HEAD".equals(method)) {
                try (OutputStream body = exchange.getResponseBody()) {
                    body.write(bytes);
                }
            }
        }
    }

    private static String contentType(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".html")) {
            return "text/html; charset=utf-8";
        }
        if (lower.endsWith(".js") || lower.endsWith(".mjs")) {
            return "text/javascript; charset=utf-8";
        }
        if (lower.endsWith(".css")) {
            return "text/css; charset=utf-8";
        }
        if (lower.endsWith(".json")) {
            return "application/json; charset=utf-8";
        }
        if (lower.endsWith(".svg")) {
            return "image/svg+xml";
        }
        if (lower.endsWith(".png")) {
            return "image/png";
        }
        if (lower.endsWith(".ico")) {
            return "image/x-icon";
        }
        if (lower.endsWith(".wasm")) {
            return "application/wasm";
        }
        return "application/octet-stream";
    }

    private static void sendJson(HttpExchange exchange, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "application/json; charset=utf-8");
        headers.set("Cache-Control", "no-store");
        try {
            exchange.sendResponseHeaders(status, bytes.length);
            try (OutputStream body = exchange.getResponseBody()) {
                body.write(bytes);
            }
        } catch (IOException ignored) {
            // The client may have aborted the request.
        }
    }

    private static String errorJson(String message) {
        return "{\"message\":" + Json.quote(message == null ? "Unknown error." : message) + "}";
    }

    private static String analysisJson(ImageResult[] images, long elapsedMs, long decodeMs,
            String backend, String deviceName, Feature feature, Comparison comparison, String message) {
        StringBuilder json = new StringBuilder("{\"images\":[");
        for (int i = 0; i < images.length; i++) {
            if (i > 0) {
                json.append(',');
            }
            ImageResult image = images[i];
            json.append("{\"width\":").append(image.width())
                    .append(",\"height\":").append(image.height())
                    .append(",\"faces\":[");
            for (int j = 0; j < image.faces().size(); j++) {
                if (j > 0) {
                    json.append(',');
                }
                json.append(faceJson(image.faces().get(j)));
            }
            json.append("]}");
        }
        json.append("],\"elapsedMs\":").append(elapsedMs)
                .append(",\"decodeMs\":").append(decodeMs)
                .append(",\"backend\":").append(Json.quote(backend))
                .append(",\"deviceName\":")
                .append(deviceName == null ? "null" : Json.quote(deviceName))
                .append(",\"feature\":");
        if (feature == null) {
            json.append("null");
        } else {
            json.append("{\"model\":").append(Json.quote(feature.model()))
                    .append(",\"pipeline\":").append(Json.quote(feature.pipeline()))
                    .append(",\"values\":[");
            for (int i = 0; i < feature.values().length; i++) {
                if (i > 0) {
                    json.append(',');
                }
                json.append(Json.number(feature.values()[i]));
            }
            json.append("]}");
        }
        json.append(",\"comparison\":")
                .append(comparison == null ? "null" : comparisonJson(comparison))
                .append(",\"message\":")
                .append(message == null ? "null" : Json.quote(message))
                .append('}');
        return json.toString();
    }

    private static String comparisonJson(Comparison comparison) {
        return "{\"similarity\":" + Json.number(comparison.similarity())
                + ",\"threshold\":" + Json.number(comparison.threshold())
                + ",\"differentThreshold\":" + Json.number(comparison.differentThreshold())
                + ",\"isMatch\":" + comparison.isMatch()
                + ",\"decision\":" + Json.quote(comparison.decision()) + "}";
    }

    private static String faceJson(DetectedFace face) {
        return "{\"x\":" + Json.number(face.x())
                + ",\"y\":" + Json.number(face.y())
                + ",\"width\":" + Json.number(face.width())
                + ",\"height\":" + Json.number(face.height())
                + ",\"score\":" + Json.number(face.score())
                + ",\"landmarks\":{"
                + "\"rightEye\":" + pointJson(face.landmarks().rightEye().x(), face.landmarks().rightEye().y())
                + ",\"leftEye\":" + pointJson(face.landmarks().leftEye().x(), face.landmarks().leftEye().y())
                + ",\"nose\":" + pointJson(face.landmarks().nose().x(), face.landmarks().nose().y())
                + ",\"rightMouth\":" + pointJson(face.landmarks().rightMouth().x(), face.landmarks().rightMouth().y())
                + ",\"leftMouth\":" + pointJson(face.landmarks().leftMouth().x(), face.landmarks().leftMouth().y())
                + "}}";
    }

    private static String pointJson(float x, float y) {
        return "{\"x\":" + Json.number(x) + ",\"y\":" + Json.number(y) + "}";
    }

    private enum PhotoMode { DETECT, COMPARE, EXTRACT }

    private record ImageResult(int width, int height, List<DetectedFace> faces) {
    }

    private record Feature(String model, String pipeline, float[] values) {
    }

    private record Comparison(float similarity, float threshold, float differentThreshold,
            boolean isMatch, String decision) {
    }

    private record GpuModels(VulkanInference device, FaceDetector detector,
            FaceRecognizer recognizer) {
    }
}
