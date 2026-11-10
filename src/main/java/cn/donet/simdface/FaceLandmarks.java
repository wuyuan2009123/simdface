package cn.donet.simdface;

/** Five landmarks in subject-relative order and original image coordinates. */
public record FaceLandmarks(
        FacePoint rightEye, FacePoint leftEye, FacePoint nose,
        FacePoint rightMouth, FacePoint leftMouth) {

    public FacePoint[] asArray() {
        return new FacePoint[] {rightEye, leftEye, nose, rightMouth, leftMouth};
    }
}
