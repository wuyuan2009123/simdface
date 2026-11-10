# SimdFace Java

[SimdFace](https://github.com/donet-org-cn/SimdFace) 的 Java 移植版：用 YuNet 检测人脸和五个关键点，用 SFace 对齐人脸并提取归一化的 128 维特征，支持 1:1 比对。

推理引擎完全用 Java 实现：项目自带一个只解析所需 ONNX 字段和算子的读取器，以及 `Conv`、`BatchNormalization`、`PRelu`、`MaxPool`、`Resize`、`Transpose`、`Sigmoid`、`Add`、`Sub`、`Mul`、`Gemm` 等 CPU 内核，不依赖 ONNX Runtime。CPU 路径零第三方依赖，只需要 JDK 17+。

## 功能

- **人脸检测**：输出置信度、边界框和五个关键点，按置信度排序并做 IoU 非极大值抑制。
- **特征提取与比对**：单次对齐裁剪（CLI）或三种裁剪加镜像共六次融合（Web / `embedForVerification`），输出 L2 归一化向量和余弦相似度。
- **CPU 与 Vulkan 后端**：CPU 内核按通道并行；Vulkan 后端通过 LWJGL 调用系统 loader，复用原版的两份 SPIR-V 计算着色器，六次裁剪可在一次提交中完成。
- **CLI**：`detect` 输出检测 JSON，`compare` 输出 `similarity`、`threshold`、`isMatch`。
- **Web 测试台**：沿用上游前端，支持单图检测、双图比对、提取特征、向量比对四种模式，HTTP API 与原版一致。
- **大图与 EXIF**：JPEG/PNG 等通过 ImageIO 解码，应用 EXIF 方向（1–8），检测坐标映射回校正后的图像。

## 与 C# 版的差异

| 项目 | 说明 |
| --- | --- |
| 浏览器 WASM / NetWasm | 未移植 |
| Vulkan 实现 | 使用 LWJGL 3 的 Vulkan 绑定，着色器与原版相同；支持 64 位 Windows 和 Linux |
| 图像解码 | 原版使用 SimdCodecs，本项目使用 JDK ImageIO（JPEG/PNG/BMP/GIF）。WebP 默认不支持 |
| 模型加载 | 保留 SHA-256 校验，校验值与原版一致 |

## 构建与运行

需要 JDK 17+ 和 Maven。Vulkan 后端还要求系统安装 Vulkan loader 和显卡驱动，并支持 64 位 Windows/Linux。

```bash
mvn -q package
```

产物为 `target/simdface.jar`，模型已放在 `models/`。

### 命令行

```bash
java -jar target/simdface.jar detect photo.jpg
```

```bash
java -jar target/simdface.jar compare first.jpg second.jpg --threshold 0.68
```

```bash
java -jar target/simdface.jar detect photo.jpg --backend vulkan
```

选项：

- `--models <dir>`：模型目录，默认 `models`
- `--score <0..1>`：检测阈值，默认 `0.8`
- `--threshold <-1..1>`：比对阈值，默认 `0.363`
- `--backend cpu|vulkan`：CPU 或 Vulkan 推理，默认 `cpu`

输出示例：

```json
{"width":512,"height":512,"faces":[{"x":207.78564,"y":182.5318,"width":145.93884,"height":206.9199,"score":0.9089786,"landmarks":{"rightEye":{"x":271.63553,"y":269.44812},"leftEye":{"x":328.47333,"y":276.09894},"nose":{"x":309.69385,"y":312.87872},"rightMouth":{"x":270.92706,"y":342.1558},"leftMouth":{"x":311.31155,"y":348.4068}}}]}
```

### Web 测试台

```bash
java -cp target/simdface.jar cn.donet.simdface.web.WebServer --port 5088
```

打开 <http://127.0.0.1:5088/>。可用参数：`--host`、`--port`、`--models`。

在「推理设备」中选择「后端 GPU（Vulkan）」即可使用显卡；GPU 设备在第一次请求时惰性初始化。若 Vulkan loader、显卡或驱动不可用，接口返回 `503` 并提示切回 CPU。

HTTP API（`multipart/form-data`，请求头 `X-SimdFace-Request: 1`）：

- `POST /api/detect`：字段 `first`，查询参数 `score`、`backend`
- `POST /api/compare`：字段 `first`、`second`，查询参数 `score`、`threshold`
- `POST /api/extract`：字段 `first`，返回 `feature`（128 维 JSON）
- `POST /api/vectors`：字段 `first`、`second` 为两个特征 JSON

同一时刻只处理一个推理请求，重复请求返回 `429`。

### 作为库调用

```java
import cn.donet.simdface.*;

FaceDetector detector = new FaceDetector("models/face_detection_yunet_2023mar.onnx");
FaceRecognizer recognizer = new FaceRecognizer("models/face_recognition_sface_2021dec.onnx");

RgbImage image = ImageDecoder.readFile("photo.jpg");
DetectedFace face = detector.detect(image.rgb(), image.width(), image.height()).get(0);
float[] embedding = recognizer.embedForVerification(
        image.rgb(), image.width(), image.height(), face.landmarks());

float similarity = FaceRecognizer.cosineSimilarity(embeddingA, embeddingB);
```

`FaceDetector` / `FaceRecognizer` 也接受模型字节数组；`RgbImage` 是紧密排列的 RGB24。使用 GPU 时创建一个共享设备并传给两个模型：

```java
try (VulkanInference vulkan = new VulkanInference()) {
    FaceDetector detector = new FaceDetector("models/face_detection_yunet_2023mar.onnx", vulkan);
    FaceRecognizer recognizer = new FaceRecognizer("models/face_recognition_sface_2021dec.onnx", vulkan);
    // ...
}
```

## 项目结构

| 路径 | 职责 |
| --- | --- |
| `src/main/java/cn/donet/simdface/ModelGraph.java` | ONNX protobuf 解析与图层调度 |
| `.../Kernels.java` | CPU 算子实现（通道并行，`-Dsimdface.threads=N` 可调） |
| `.../FaceDetector.java` | YuNet 前处理、解码与 NMS |
| `.../FaceRecognizer.java` | 五点对齐、六裁剪特征融合与余弦相似度 |
| `.../ImageDecoder.java` | ImageIO 解码与 EXIF 方向校正 |
| `.../VulkanInference.java` | Vulkan 设备、缓冲区、描述符与命令提交（LWJGL） |
| `.../GraphPlan.java`、`.../Dispatch.java` | Vulkan 张量布局与计算调度计划 |
| `.../Cli.java` | `detect` / `compare` 命令 |
| `.../web/WebServer.java` | HTTP API 与静态资源 |
| `src/main/resources/shaders/` | 原版 `tensor.spv`、`pointwise.spv` |
| `src/main/resources/wwwroot/` | 上游 Web 前端 |

## 与 C# 原版的一致性验证

使用同一份 RGB 像素输入（绕过 JPEG 解码差异）时：

- 检测框与五个关键点**逐位一致**；
- 128 维特征最大绝对误差约 `5.1e-7`，余弦相似度 `0.99999988`；
- PNG 无损路径下，Java 与 C# 的检测 JSON 完全相同；
- Vulkan 路径下，Java 与 C# 在同一 PNG 上的检测 JSON 也**逐位一致**；
- EXIF 方向 2（水平镜像）和 6（旋转 + 宽高交换）下的输出尺寸与检测坐标一致。

JPEG 解码由不同解码器完成，像素会有细微差异，从而让检测框和分数出现末位差别；上游 README 也说明浏览器与后端 JPEG 路径存在类似现象。Vulkan 与 CPU 由于浮点累加顺序不同，特征也会有约 `10^-6` 量级的差异，不影响阈值判断。

本机 Intel UHD Graphics（Vulkan 1.2）与 8 核 CPU 实测：单图检测 CPU 约 0.1–0.3 秒、Vulkan 约 0.15–0.2 秒；双图六裁剪比对 CPU 约 2.2 秒、Vulkan 约 1.0 秒。CPU 线程数可用 `-Dsimdface.threads=N` 调整。

## 模型与许可证

| 权重 | 用途 | 上游和许可证 |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | 人脸检测和五点定位 | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) · MIT |
| `face_recognition_sface_2021dec.onnx` | 128 维人脸特征 | [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) · Apache-2.0 |

Java 移植源码沿用上游的 [MIT 许可证](LICENSE)。模型许可证见 `models/` 下的文本文件。
