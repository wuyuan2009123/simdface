package cn.donet.simdface;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

/** Decodes JPEG/PNG (and other ImageIO formats) into orientation-corrected RGB24. */
public final class ImageDecoder {

    public static final int MaxFileBytes = 64 * 1024 * 1024;
    public static final int DefaultMaxPixels = 64_000_000;

    static {
        ImageIO.setUseCache(false);
    }

    private ImageDecoder() {
    }

    public static RgbImage readFile(String path) throws IOException {
        java.io.File file = new java.io.File(path);
        if (file.length() > MaxFileBytes) {
            throw new InvalidDataException(
                    "Image file exceeds " + MaxFileBytes / (1024 * 1024) + " MiB: " + path);
        }
        return decode(Files.readAllBytes(Path.of(path)), path);
    }

    public static RgbImage decode(byte[] data, String label) {
        if (data.length > MaxFileBytes) {
            throw new InvalidDataException(
                    "Image file exceeds " + MaxFileBytes / (1024 * 1024) + " MiB: " + label);
        }
        int orientation = isJpeg(data) ? readJpegOrientation(data) : 1;
        BufferedImage image;
        int width;
        int height;
        try (ImageInputStream stream = ImageIO.createImageInputStream(new ByteArrayInputStream(data))) {
            if (stream == null) {
                throw new InvalidDataException("Unsupported or invalid image: " + label);
            }
            Iterator<ImageReader> readers = ImageIO.getImageReaders(stream);
            if (!readers.hasNext()) {
                throw new InvalidDataException("Unsupported or invalid image: " + label);
            }
            ImageReader reader = readers.next();
            try {
                reader.setInput(stream, true, true);
                width = reader.getWidth(0);
                height = reader.getHeight(0);
                int outputWidth = orientation >= 5 && orientation <= 8 ? height : width;
                int outputHeight = orientation >= 5 && orientation <= 8 ? width : height;
                validateDimensions(outputWidth, outputHeight, label);
                image = reader.read(0);
                if (image == null) {
                    throw new InvalidDataException("Could not decode image: " + label);
                }
            } finally {
                reader.dispose();
            }
        } catch (IOException | RuntimeException error) {
            if (error instanceof InvalidDataException invalid) {
                throw invalid;
            }
            throw new InvalidDataException("Could not decode image: " + label, error);
        }

        return toRgb(image, orientation, label);
    }

    private static void validateDimensions(int width, int height, String label) {
        long pixels = (long) width * height;
        if (width <= 0 || height <= 0 || pixels > DefaultMaxPixels || pixels > Integer.MAX_VALUE / 3) {
            throw new InvalidDataException(
                    "Image must contain between 1 and " + DefaultMaxPixels + " pixels: " + label);
        }
    }

    private static RgbImage toRgb(BufferedImage image, int orientation, String label) {
        int sourceWidth = image.getWidth();
        int sourceHeight = image.getHeight();
        boolean swapped = orientation >= 5 && orientation <= 8;
        int outWidth = swapped ? sourceHeight : sourceWidth;
        int outHeight = swapped ? sourceWidth : sourceHeight;
        validateDimensions(outWidth, outHeight, label);

        byte[] rgb = new byte[outWidth * outHeight * 3];
        int[] row = new int[sourceWidth];
        for (int y = 0; y < sourceHeight; y++) {
            image.getRGB(0, y, sourceWidth, 1, row, 0, sourceWidth);
            for (int x = 0; x < sourceWidth; x++) {
                int dx;
                int dy;
                switch (orientation) {
                    case 2 -> {
                        dx = sourceWidth - 1 - x;
                        dy = y;
                    }
                    case 3 -> {
                        dx = sourceWidth - 1 - x;
                        dy = sourceHeight - 1 - y;
                    }
                    case 4 -> {
                        dx = x;
                        dy = sourceHeight - 1 - y;
                    }
                    case 5 -> {
                        dx = y;
                        dy = x;
                    }
                    case 6 -> {
                        dx = sourceHeight - 1 - y;
                        dy = x;
                    }
                    case 7 -> {
                        dx = sourceHeight - 1 - y;
                        dy = sourceWidth - 1 - x;
                    }
                    case 8 -> {
                        dx = y;
                        dy = sourceWidth - 1 - x;
                    }
                    default -> {
                        dx = x;
                        dy = y;
                    }
                }
                int argb = row[x];
                int index = (dy * outWidth + dx) * 3;
                rgb[index] = (byte) (argb >> 16);
                rgb[index + 1] = (byte) (argb >> 8);
                rgb[index + 2] = (byte) argb;
            }
        }
        return new RgbImage(rgb, outWidth, outHeight);
    }

    private static boolean isJpeg(byte[] data) {
        return data.length >= 3 && (data[0] & 0xFF) == 0xFF && (data[1] & 0xFF) == 0xD8
                && (data[2] & 0xFF) == 0xFF;
    }

    private static int readJpegOrientation(byte[] data) {
        int length = data.length;
        int position = 2;
        while (position + 4 <= length) {
            if ((data[position] & 0xFF) != 0xFF) {
                position++;
                continue;
            }
            while (position < length && (data[position] & 0xFF) == 0xFF) {
                position++;
            }
            if (position >= length) {
                break;
            }
            int marker = data[position++] & 0xFF;
            if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
                continue;
            }
            if (marker == 0xD9 || marker == 0xDA) {
                break;
            }
            if (position + 2 > length) {
                break;
            }
            int segmentLength = ((data[position] & 0xFF) << 8) | (data[position + 1] & 0xFF);
            if (segmentLength < 2 || position + segmentLength > length) {
                break;
            }
            int payload = position + 2;
            if (marker == 0xE1 && segmentLength >= 8
                    && data[payload] == 'E' && data[payload + 1] == 'x'
                    && data[payload + 2] == 'i' && data[payload + 3] == 'f'
                    && data[payload + 4] == 0 && data[payload + 5] == 0) {
                int orientation = readTiffOrientation(data, payload + 6, position + segmentLength);
                if (orientation >= 1 && orientation <= 8) {
                    return orientation;
                }
            }
            position += segmentLength;
        }
        return 1;
    }

    private static int readTiffOrientation(byte[] data, int tiffStart, int end) {
        if (tiffStart < 0 || tiffStart + 8 > end) {
            return 1;
        }
        boolean little;
        if (data[tiffStart] == 'I' && data[tiffStart + 1] == 'I') {
            little = true;
        } else if (data[tiffStart] == 'M' && data[tiffStart + 1] == 'M') {
            little = false;
        } else {
            return 1;
        }
        if (readU16(data, tiffStart + 2, little) != 42) {
            return 1;
        }
        long ifdOffset = readU32(data, tiffStart + 4, little);
        long ifd = tiffStart + ifdOffset;
        if (ifd < tiffStart || ifd + 2 > end) {
            return 1;
        }
        int count = readU16(data, (int) ifd, little);
        for (int i = 0; i < count; i++) {
            long entry = ifd + 2 + (long) i * 12;
            if (entry + 12 > end) {
                return 1;
            }
            int tag = readU16(data, (int) entry, little);
            if (tag == 0x0112) {
                int type = readU16(data, (int) entry + 2, little);
                long valueCount = readU32(data, (int) entry + 4, little);
                if (type == 3 && valueCount >= 1) {
                    int value = readU16(data, (int) entry + 8, little);
                    return value >= 1 && value <= 8 ? value : 1;
                }
                return 1;
            }
        }
        return 1;
    }

    private static int readU16(byte[] data, int offset, boolean little) {
        int first = data[offset] & 0xFF;
        int second = data[offset + 1] & 0xFF;
        return little ? first | second << 8 : first << 8 | second;
    }

    private static long readU32(byte[] data, int offset, boolean little) {
        long first = data[offset] & 0xFFL;
        long second = data[offset + 1] & 0xFFL;
        long third = data[offset + 2] & 0xFFL;
        long fourth = data[offset + 3] & 0xFFL;
        return little
                ? first | second << 8 | third << 16 | fourth << 24
                : first << 24 | second << 16 | third << 8 | fourth;
    }
}
