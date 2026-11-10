package cn.donet.simdface;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.vulkan.VK10.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.IdentityHashMap;
import java.util.Locale;
import java.util.Map;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkApplicationInfo;
import org.lwjgl.vulkan.VkBufferCopy;
import org.lwjgl.vulkan.VkBufferCreateInfo;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorBufferInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo;
import org.lwjgl.vulkan.VkFenceCreateInfo;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkInstanceCreateInfo;
import org.lwjgl.vulkan.VkMemoryAllocateInfo;
import org.lwjgl.vulkan.VkMemoryBarrier;
import org.lwjgl.vulkan.VkMemoryRequirements;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkQueueFamilyProperties;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

/**
 * Reusable Vulkan compute device for YuNet and SFace inference. Close it after both models
 * finish. Ported from the C# Vulkan backend and the project's embedded SPIR-V shaders.
 */
public final class VulkanInference implements AutoCloseable {

    private static final long MAX_STORAGE_BYTES = 128L * 1024 * 1024;
    private static final int STAGING_FLAGS =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    private final Object gate = new Object();
    private final Map<ModelGraph, ModelState> models = new IdentityHashMap<>();
    private final String deviceName;

    private VkInstance instance;
    private VkDevice device;
    private VkQueue queue;
    private VkCommandBuffer command;
    private long commandPool;
    private long fence;
    private long descriptorLayout;
    private long pipelineLayout;
    private long tensorPipeline;
    private long pointwisePipeline;
    private int[] memoryTypeFlags;
    private boolean closed;

    public VulkanInference() {
        String name;
        try {
            name = initialize();
        } catch (LinkageError error) {
            closeQuietly();
            throw new UnsupportedOperationException("The system Vulkan loader is unavailable.", error);
        } catch (RuntimeException error) {
            closeQuietly();
            throw error;
        }
        deviceName = name;
    }

    public String deviceName() {
        return deviceName;
    }

    private String initialize() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch", "");
        if (!(os.contains("win") || os.contains("linux")) || !arch.contains("64")) {
            throw new UnsupportedOperationException(
                    "The Vulkan backend supports 64-bit Windows and Linux.");
        }

        int queueFamily;
        String name;
        try (MemoryStack stack = stackPush()) {
            var appInfo = VkApplicationInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
                    .pApplicationName(stack.UTF8("SimdFace"))
                    .applicationVersion(VK_MAKE_VERSION(0, 1, 0))
                    .pEngineName(stack.UTF8("SimdFace"))
                    .engineVersion(VK_MAKE_VERSION(0, 1, 0))
                    .apiVersion(VK_API_VERSION_1_0);
            var createInfo = VkInstanceCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
                    .pApplicationInfo(appInfo);
            PointerBuffer pInstance = stack.mallocPointer(1);
            check(vkCreateInstance(createInfo, null, pInstance), "create instance");
            instance = new VkInstance(pInstance.get(0), createInfo);

            IntBuffer count = stack.mallocInt(1);
            check(vkEnumeratePhysicalDevices(instance, count, null), "enumerate devices");
            if (count.get(0) == 0) {
                throw new UnsupportedOperationException("No Vulkan device was found.");
            }
            PointerBuffer devices = stack.mallocPointer(count.get(0));
            check(vkEnumeratePhysicalDevices(instance, count, devices), "enumerate devices");

            VkPhysicalDevice best = null;
            int bestRank = 0;
            int bestFamily = 0;
            String bestName = "";
            VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.malloc(stack);
            for (int i = 0; i < devices.capacity(); i++) {
                VkPhysicalDevice candidate = new VkPhysicalDevice(devices.get(i), instance);
                vkGetPhysicalDeviceProperties(candidate, properties);
                int rank = switch (properties.deviceType()) {
                    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU -> 3;
                    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU -> 2;
                    case VK_PHYSICAL_DEVICE_TYPE_CPU -> 0;
                    default -> 1;
                };
                if (rank <= bestRank) {
                    continue;
                }
                IntBuffer familyCount = stack.mallocInt(1);
                vkGetPhysicalDeviceQueueFamilyProperties(candidate, familyCount, null);
                VkQueueFamilyProperties.Buffer families =
                        VkQueueFamilyProperties.malloc(familyCount.get(0), stack);
                vkGetPhysicalDeviceQueueFamilyProperties(candidate, familyCount, families);
                for (int j = 0; j < families.capacity(); j++) {
                    if (families.get(j).queueCount() > 0
                            && (families.get(j).queueFlags() & VK_QUEUE_COMPUTE_BIT) != 0) {
                        best = candidate;
                        bestFamily = j;
                        bestName = properties.deviceNameString();
                        bestRank = rank;
                        break;
                    }
                }
            }
            if (best == null) {
                throw new UnsupportedOperationException(
                        "No hardware Vulkan device with a compute queue was found.");
            }
            name = bestName;
            queueFamily = bestFamily;

            VkPhysicalDeviceMemoryProperties memoryProperties =
                    VkPhysicalDeviceMemoryProperties.malloc(stack);
            vkGetPhysicalDeviceMemoryProperties(best, memoryProperties);
            memoryTypeFlags = new int[memoryProperties.memoryTypeCount()];
            for (int i = 0; i < memoryTypeFlags.length; i++) {
                memoryTypeFlags[i] = memoryProperties.memoryTypes(i).propertyFlags();
            }

            FloatBuffer priorities = stack.floats(1f);
            VkDeviceQueueCreateInfo.Buffer queueInfos = VkDeviceQueueCreateInfo.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                    .queueFamilyIndex(queueFamily)
                    .pQueuePriorities(priorities);
            VkDeviceCreateInfo deviceInfo = VkDeviceCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                    .pQueueCreateInfos(queueInfos);
            PointerBuffer pDevice = stack.mallocPointer(1);
            check(vkCreateDevice(best, deviceInfo, null, pDevice), "create device");
            device = new VkDevice(pDevice.get(0), best, deviceInfo);

            PointerBuffer pQueue = stack.mallocPointer(1);
            vkGetDeviceQueue(device, queueFamily, 0, pQueue);
            queue = new VkQueue(pQueue.get(0), device);

            VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                    .queueFamilyIndex(queueFamily);
            LongBuffer pHandle = stack.mallocLong(1);
            check(vkCreateCommandPool(device, poolInfo, null, pHandle), "create command pool");
            commandPool = pHandle.get(0);

            VkCommandBufferAllocateInfo allocate = VkCommandBufferAllocateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                    .commandPool(commandPool)
                    .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                    .commandBufferCount(1);
            check(vkAllocateCommandBuffers(device, allocate, pDevice), "allocate command buffer");
            command = new VkCommandBuffer(pDevice.get(0), device);

            VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
            check(vkCreateFence(device, fenceInfo, null, pHandle), "create fence");
            fence = pHandle.get(0);

            createLayouts(stack);
            tensorPipeline = createPipeline("tensor", stack);
            pointwisePipeline = createPipeline("pointwise", stack);
        }
        return name;
    }

    private void createLayouts(MemoryStack stack) {
        VkDescriptorSetLayoutBinding.Buffer binding = VkDescriptorSetLayoutBinding.calloc(1, stack)
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                .pBindings(binding);
        LongBuffer pHandle = stack.mallocLong(1);
        check(vkCreateDescriptorSetLayout(device, layoutInfo, null, pHandle),
                "create descriptor layout");
        descriptorLayout = pHandle.get(0);

        VkPushConstantRange.Buffer push = VkPushConstantRange.calloc(1, stack)
                .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)
                .offset(0)
                .size(Dispatch.SIZE);
        VkPipelineLayoutCreateInfo pipelineInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                .pSetLayouts(stack.longs(descriptorLayout))
                .pPushConstantRanges(push);
        check(vkCreatePipelineLayout(device, pipelineInfo, null, pHandle),
                "create pipeline layout");
        pipelineLayout = pHandle.get(0);
    }

    private long createPipeline(String name, MemoryStack stack) {
        byte[] bytes = readResource("/shaders/" + name + ".spv");
        ByteBuffer code = stack.malloc(bytes.length);
        code.put(bytes).flip();
        VkShaderModuleCreateInfo moduleInfo = VkShaderModuleCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
                .pCode(code);
        LongBuffer pHandle = stack.mallocLong(1);
        check(vkCreateShaderModule(device, moduleInfo, null, pHandle), "create shader module");
        long module = pHandle.get(0);
        try {
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                    .stage(VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(module)
                    .pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer infos = VkComputePipelineCreateInfo.calloc(1, stack);
            infos.get(0)
                    .sType(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO)
                    .stage(stage)
                    .layout(pipelineLayout)
                    .basePipelineHandle(NULL)
                    .basePipelineIndex(-1);
            check(vkCreateComputePipelines(device, NULL, infos, null, pHandle),
                    "create " + name + " pipeline");
            return pHandle.get(0);
        } finally {
            vkDestroyShaderModule(device, module, null);
        }
    }

    private static byte[] readResource(String name) {
        try (InputStream input = VulkanInference.class.getResourceAsStream(name)) {
            if (input == null) {
                throw new IllegalStateException("Missing Vulkan shader: " + name);
            }
            return input.readAllBytes();
        } catch (IOException error) {
            throw new IllegalStateException("Could not read Vulkan shader: " + name, error);
        }
    }

    float[][] run(ModelGraph graph, Tensor input) {
        return runBatch(graph, new Tensor[] {input});
    }

    float[][] runBatch(ModelGraph graph, Tensor[] inputs) {
        synchronized (gate) {
            if (closed) {
                throw new IllegalStateException("VulkanInference has been closed.");
            }
            Tensor first = inputs[0];
            ModelState state = models.computeIfAbsent(graph, ignored -> new ModelState());
            GraphPlan plan = state.plan;
            if (plan == null || plan.input().channels() != first.channels()
                    || plan.input().height() != first.height()
                    || plan.input().width() != first.width()) {
                plan = graph.planVulkan(first);
                state.plan = plan;
            }

            int inputLength = first.data().length;
            long outputLength = 0;
            for (TensorSlot output : plan.outputs()) {
                outputLength += output.length();
            }
            long inputTotal = (long) inputLength * inputs.length;
            long outputStart = inputs.length == 1 ? 0 : inputTotal;
            long stagingFloats = Math.max(plan.weightLength(),
                    Math.max(inputTotal, outputStart + outputLength * inputs.length));
            if (stagingFloats * 4 > Integer.MAX_VALUE - 8) {
                throw new UnsupportedOperationException("The Vulkan staging buffer is too large.");
            }
            int stagingLength = (int) stagingFloats;

            Arena arena = state.arena;
            if (arena == null || arena.storage.bytes < (long) plan.length() * 4
                    || arena.staging.bytes < (long) stagingLength * 4) {
                state.arena = null;
                if (arena != null) {
                    arena.close();
                }
                arena = createArena(plan, stagingLength);
                state.arena = arena;
            }

            FloatBuffer staging = arena.stagingFloats();
            for (int i = 0; i < inputs.length; i++) {
                Tensor sample = inputs[i];
                if (sample.channels() != first.channels() || sample.height() != first.height()
                        || sample.width() != first.width()
                        || sample.data().length != inputLength) {
                    throw new IllegalArgumentException("Vulkan batch inputs must share the same shape.");
                }
                staging.position(i * inputLength);
                staging.put(sample.data());
            }

            begin();
            int outputOffset = (int) outputStart;
            try (MemoryStack stack = stackPush()) {
                ByteBuffer push = stack.malloc(Dispatch.SIZE).order(ByteOrder.LITTLE_ENDIAN);
                long set = arena.descriptorSet;
                vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                        0, stack.longs(set), null);
                long boundPipeline = NULL;
                for (int i = 0; i < inputs.length; i++) {
                    copy(arena.staging, arena.storage, i * inputLength, plan.input().offset(), inputLength);
                    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                    for (Dispatch dispatch : plan.dispatches()) {
                        long pipeline = dispatch.op == 13 ? pointwisePipeline : tensorPipeline;
                        if (pipeline != boundPipeline) {
                            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                            boundPipeline = pipeline;
                        }
                        dispatch.writePushConstants(push);
                        vkCmdPushConstants(command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                                0, push);
                        int plane = dispatch.outHeight * dispatch.outWidth;
                        if (dispatch.op == 13) {
                            vkCmdDispatch(command, (plane + 15) / 16, (dispatch.outChannels + 7) / 8, 1);
                        } else if (dispatch.op == 12) {
                            vkCmdDispatch(command, dispatch.outChannels, 1, 1);
                        } else {
                            vkCmdDispatch(command, (dispatch.outChannels * plane + 63) / 64, 1, 1);
                        }
                        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                    }

                    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                    for (TensorSlot output : plan.outputs()) {
                        copy(arena.storage, arena.staging, output.offset(), outputOffset, output.length());
                        outputOffset += output.length();
                    }
                    if (i + 1 < inputs.length) {
                        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                    }
                }
                barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            }
            submit();

            float[][] results = new float[inputs.length * plan.outputs().length][];
            outputOffset = (int) outputStart;
            staging = arena.stagingFloats();
            for (int i = 0; i < results.length; i++) {
                int length = plan.outputs()[i % plan.outputs().length].length();
                float[] result = new float[length];
                staging.position(outputOffset);
                staging.get(result);
                results[i] = result;
                outputOffset += length;
            }
            return results;
        }
    }

    private Arena createArena(GraphPlan plan, int stagingLength) {
        long storageBytes = (long) plan.length() * 4;
        if (storageBytes > MAX_STORAGE_BYTES) {
            throw new UnsupportedOperationException("The model exceeds the Vulkan tensor storage limit.");
        }
        Arena arena = new Arena();
        try {
            arena.storage = createBuffer(storageBytes,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            arena.staging = createBuffer((long) stagingLength * 4,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    STAGING_FLAGS);
            try (MemoryStack stack = stackPush()) {
                VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(1, stack)
                        .type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1);
                VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                        .maxSets(1)
                        .pPoolSizes(sizes);
                LongBuffer pHandle = stack.mallocLong(1);
                check(vkCreateDescriptorPool(device, poolInfo, null, pHandle),
                        "create descriptor pool");
                arena.descriptorPool = pHandle.get(0);

                VkDescriptorSetAllocateInfo setInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                        .descriptorPool(arena.descriptorPool)
                        .pSetLayouts(stack.longs(descriptorLayout));
                check(vkAllocateDescriptorSets(device, setInfo, pHandle),
                        "allocate descriptor set");
                arena.descriptorSet = pHandle.get(0);

                VkDescriptorBufferInfo.Buffer bufferInfo = VkDescriptorBufferInfo.calloc(1, stack)
                        .buffer(arena.storage.handle)
                        .offset(0)
                        .range(arena.storage.bytes);
                VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack)
                        .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                        .dstSet(arena.descriptorSet)
                        .dstBinding(0)
                        .descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(bufferInfo);
                vkUpdateDescriptorSets(device, write, null);
            }

            FloatBuffer staging = arena.stagingFloats();
            int offset = 0;
            for (float[] weights : plan.weights()) {
                staging.position(offset);
                staging.put(weights);
                offset += weights.length;
            }

            begin();
            copy(arena.staging, arena.storage, 0, 0, plan.weightLength());
            barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            submit();
            return arena;
        } catch (RuntimeException error) {
            arena.close();
            throw error;
        }
    }

    private GpuBuffer createBuffer(long size, int usage, int requiredFlags) {
        try (MemoryStack stack = stackPush()) {
            VkBufferCreateInfo info = VkBufferCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                    .size(size)
                    .usage(usage)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
            LongBuffer pHandle = stack.mallocLong(1);
            check(vkCreateBuffer(device, info, null, pHandle), "create buffer");
            long handle = pHandle.get(0);
            long memory = NULL;
            try {
                VkMemoryRequirements requirements = VkMemoryRequirements.malloc(stack);
                vkGetBufferMemoryRequirements(device, handle, requirements);
                int memoryType = findMemory(requirements.memoryTypeBits(), requiredFlags);
                VkMemoryAllocateInfo allocate = VkMemoryAllocateInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                        .allocationSize(requirements.size())
                        .memoryTypeIndex(memoryType);
                check(vkAllocateMemory(device, allocate, null, pHandle), "allocate buffer memory");
                memory = pHandle.get(0);
                check(vkBindBufferMemory(device, handle, memory, 0), "bind buffer memory");
                FloatBuffer mapped = null;
                if ((requiredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
                    PointerBuffer pData = stack.mallocPointer(1);
                    check(vkMapMemory(device, memory, 0, size, 0, pData), "map staging memory");
                    mapped = MemoryUtil.memFloatBuffer(pData.get(0), (int) (size / 4));
                }
                return new GpuBuffer(handle, memory, size, mapped);
            } catch (RuntimeException error) {
                if (handle != NULL) {
                    vkDestroyBuffer(device, handle, null);
                }
                if (memory != NULL) {
                    vkFreeMemory(device, memory, null);
                }
                throw error;
            }
        }
    }

    private int findMemory(int typeBits, int requiredFlags) {
        for (int i = 0; i < memoryTypeFlags.length; i++) {
            if ((typeBits & (1 << i)) != 0 && (memoryTypeFlags[i] & requiredFlags) == requiredFlags) {
                return i;
            }
        }
        throw new UnsupportedOperationException(
                "No compatible Vulkan memory type for flags " + requiredFlags + ".");
    }

    private void begin() {
        check(vkResetCommandPool(device, commandPool, 0), "reset command pool");
        try (MemoryStack stack = stackPush()) {
            VkCommandBufferBeginInfo info = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            check(vkBeginCommandBuffer(command, info), "begin command buffer");
        }
    }

    private void submit() {
        check(vkEndCommandBuffer(command), "end command buffer");
        try (MemoryStack stack = stackPush()) {
            PointerBuffer commands = stack.mallocPointer(1);
            commands.put(0, command.address());
            VkSubmitInfo info = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(commands);
            check(vkResetFences(device, fence), "reset fence");
            check(vkQueueSubmit(queue, info, fence), "submit graph");
            check(vkWaitForFences(device, fence, true, ~0L), "wait for graph");
        }
    }

    private void copy(GpuBuffer source, GpuBuffer target, int sourceOffset, int targetOffset,
            int length) {
        try (MemoryStack stack = stackPush()) {
            VkBufferCopy.Buffer region = VkBufferCopy.calloc(1, stack)
                    .srcOffset((long) sourceOffset * 4)
                    .dstOffset((long) targetOffset * 4)
                    .size((long) length * 4);
            vkCmdCopyBuffer(command, source.handle, target.handle, region);
        }
    }

    private void barrier(int sourceStage, int targetStage, int sourceAccess, int targetAccess) {
        try (MemoryStack stack = stackPush()) {
            VkMemoryBarrier.Buffer barrier = VkMemoryBarrier.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_MEMORY_BARRIER)
                    .srcAccessMask(sourceAccess)
                    .dstAccessMask(targetAccess);
            vkCmdPipelineBarrier(command, sourceStage, targetStage, 0, barrier, null, null);
        }
    }

    private static void check(int result, String operation) {
        if (result != VK_SUCCESS) {
            throw new IllegalStateException(
                    "Vulkan failed to " + operation + " (VkResult " + result + ").");
        }
    }

    private void closeQuietly() {
        try {
            close();
        } catch (Throwable ignored) {
            // Initialization failed part-way; resources are best-effort.
        }
    }

    @Override
    public void close() {
        synchronized (gate) {
            if (closed) {
                return;
            }
            closed = true;
            if (device != null) {
                vkDeviceWaitIdle(device);
                for (ModelState state : models.values()) {
                    if (state.arena != null) {
                        state.arena.close();
                    }
                }
                models.clear();
                if (pointwisePipeline != NULL) {
                    vkDestroyPipeline(device, pointwisePipeline, null);
                }
                if (tensorPipeline != NULL) {
                    vkDestroyPipeline(device, tensorPipeline, null);
                }
                if (pipelineLayout != NULL) {
                    vkDestroyPipelineLayout(device, pipelineLayout, null);
                }
                if (descriptorLayout != NULL) {
                    vkDestroyDescriptorSetLayout(device, descriptorLayout, null);
                }
                if (fence != NULL) {
                    vkDestroyFence(device, fence, null);
                }
                if (commandPool != NULL) {
                    vkDestroyCommandPool(device, commandPool, null);
                }
                vkDestroyDevice(device, null);
                device = null;
            }
            if (instance != null) {
                vkDestroyInstance(instance, null);
                instance = null;
            }
        }
    }

    private static final class ModelState {
        GraphPlan plan;
        Arena arena;
    }

    private final class Arena implements AutoCloseable {
        GpuBuffer storage;
        GpuBuffer staging;
        long descriptorPool;
        long descriptorSet;

        FloatBuffer stagingFloats() {
            return staging.mapped;
        }

        @Override
        public void close() {
            if (descriptorPool != NULL) {
                vkDestroyDescriptorPool(device, descriptorPool, null);
                descriptorPool = NULL;
            }
            destroyBuffer(staging);
            destroyBuffer(storage);
            staging = null;
            storage = null;
        }
    }

    private static final class GpuBuffer {
        final long handle;
        final long memory;
        final long bytes;
        final FloatBuffer mapped;

        GpuBuffer(long handle, long memory, long bytes, FloatBuffer mapped) {
            this.handle = handle;
            this.memory = memory;
            this.bytes = bytes;
            this.mapped = mapped;
        }
    }

    private void destroyBuffer(GpuBuffer buffer) {
        if (buffer == null) {
            return;
        }
        if (buffer.mapped != null) {
            vkUnmapMemory(device, buffer.memory);
        }
        if (buffer.handle != NULL) {
            vkDestroyBuffer(device, buffer.handle, null);
        }
        if (buffer.memory != NULL) {
            vkFreeMemory(device, buffer.memory, null);
        }
    }
}
