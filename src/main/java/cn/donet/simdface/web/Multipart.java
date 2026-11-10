package cn.donet.simdface.web;

import cn.donet.simdface.InvalidDataException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Minimal multipart/form-data parser for the web API's file fields. */
final class Multipart {

    record Part(String name, byte[] data) {
    }

    private static final Pattern NAME = Pattern.compile("name=\"([^\"]*)\"", Pattern.CASE_INSENSITIVE);
    private static final Pattern FILE_NAME = Pattern.compile("filename=\"([^\"]*)\"",
            Pattern.CASE_INSENSITIVE);
    private static final byte[] HEADER_END = {'\r', '\n', '\r', '\n'};

    private Multipart() {
    }

    static List<Part> parse(byte[] body, String boundary, int expectedCount, int fileLimit) {
        byte[] delimiter = ("--" + boundary).getBytes(StandardCharsets.ISO_8859_1);
        byte[] partSeparator = ("\r\n--" + boundary).getBytes(StandardCharsets.ISO_8859_1);
        List<Part> parts = new ArrayList<>();
        int position = indexOf(body, delimiter, 0);
        while (position >= 0) {
            int after = position + delimiter.length;
            if (after + 1 < body.length && body[after] == '-' && body[after + 1] == '-') {
                break;
            }
            if (after + 1 >= body.length || body[after] != '\r' || body[after + 1] != '\n') {
                throw new InvalidDataException("文件上传格式无效。");
            }
            int headerStart = after + 2;
            int headerEnd = indexOf(body, HEADER_END, headerStart);
            if (headerEnd < 0) {
                throw new InvalidDataException("文件上传格式无效。");
            }
            int contentStart = headerEnd + HEADER_END.length;
            int next = indexOf(body, partSeparator, contentStart);
            if (next < 0) {
                throw new InvalidDataException("文件上传格式无效。");
            }
            String headers = new String(body, headerStart, headerEnd - headerStart,
                    StandardCharsets.ISO_8859_1);
            Matcher name = NAME.matcher(headers);
            Matcher fileName = FILE_NAME.matcher(headers);
            if (!name.find() || !fileName.find()) {
                throw new InvalidDataException("上传的文件数量不正确。");
            }
            int length = next - contentStart;
            if (length > fileLimit) {
                String limit = fileLimit >= 1024 * 1024
                        ? (fileLimit / (1024 * 1024)) + " MiB"
                        : (fileLimit / 1024) + " KiB";
                throw new InvalidDataException(
                        "无法读取文件 " + (parts.size() + 1) + "。单个文件上限为 " + limit + "。");
            }
            parts.add(new Part(name.group(1), Arrays.copyOfRange(body, contentStart, next)));
            if (parts.size() > expectedCount) {
                throw new InvalidDataException("上传的文件数量不正确。");
            }
            position = next + 2;
            if (!matches(body, position, delimiter)) {
                throw new InvalidDataException("文件上传格式无效。");
            }
        }
        if (parts.size() != expectedCount) {
            throw new InvalidDataException("请选择 " + expectedCount + " 个文件。");
        }
        return parts;
    }

    private static boolean matches(byte[] data, int offset, byte[] pattern) {
        if (offset < 0 || offset + pattern.length > data.length) {
            return false;
        }
        for (int i = 0; i < pattern.length; i++) {
            if (data[offset + i] != pattern[i]) {
                return false;
            }
        }
        return true;
    }

    private static int indexOf(byte[] data, byte[] pattern, int from) {
        if (from < 0) {
            from = 0;
        }
        int last = data.length - pattern.length;
        outer:
        for (int i = from; i <= last; i++) {
            for (int j = 0; j < pattern.length; j++) {
                if (data[i + j] != pattern[j]) {
                    continue outer;
                }
            }
            return i;
        }
        return -1;
    }
}
