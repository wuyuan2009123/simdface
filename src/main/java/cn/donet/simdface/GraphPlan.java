package cn.donet.simdface;

/** Tensor offsets and compute dispatches for one input shape. */
record GraphPlan(TensorSlot input, TensorSlot[] outputs, Dispatch[] dispatches,
        float[][] weights, int weightLength, int length) {
}
