package cn.donet.simdface;

public record DetectedFace(float x, float y, float width, float height, float score, FaceLandmarks landmarks) {
}
