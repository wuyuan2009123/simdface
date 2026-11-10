package cn.donet.simdface;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Small JSON writer helpers and parser used for feature vectors and API responses. */
public final class Json {

    private Json() {
    }

    public static String quote(String value) {
        StringBuilder builder = new StringBuilder(value.length() + 2).append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> builder.append("\\\"");
                case '\\' -> builder.append("\\\\");
                case '\b' -> builder.append("\\b");
                case '\f' -> builder.append("\\f");
                case '\n' -> builder.append("\\n");
                case '\r' -> builder.append("\\r");
                case '\t' -> builder.append("\\t");
                default -> {
                    if (c < 0x20) {
                        builder.append(String.format("\\u%04x", (int) c));
                    } else {
                        builder.append(c);
                    }
                }
            }
        }
        return builder.append('"').toString();
    }

    public static String number(float value) {
        return Float.isFinite(value) ? Float.toString(value) : "null";
    }

    public static String number(double value) {
        return Double.isFinite(value) ? Double.toString(value) : "null";
    }

    public static Map<String, Object> parseObject(String text) {
        Parser parser = new Parser(text);
        Object value = parser.parseDocument();
        if (value instanceof Map<?, ?> map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> result = (Map<String, Object>) map;
            return result;
        }
        throw new InvalidDataException("Invalid JSON: expected an object.");
    }

    private static final class Parser {
        private final String text;
        private int position;

        Parser(String text) {
            this.text = text;
        }

        Object parseDocument() {
            Object value = parseValue();
            skipWhitespace();
            if (position != text.length()) {
                throw error();
            }
            return value;
        }

        private Object parseValue() {
            skipWhitespace();
            if (position >= text.length()) {
                throw error();
            }
            char c = text.charAt(position);
            return switch (c) {
                case '{' -> parseObjectValue();
                case '[' -> parseArray();
                case '"' -> parseString();
                case 't' -> parseLiteral("true", Boolean.TRUE);
                case 'f' -> parseLiteral("false", Boolean.FALSE);
                case 'n' -> parseLiteral("null", null);
                default -> parseNumber();
            };
        }

        private Map<String, Object> parseObjectValue() {
            Map<String, Object> result = new LinkedHashMap<>();
            position++;
            skipWhitespace();
            if (peek('}')) {
                position++;
                return result;
            }
            while (true) {
                skipWhitespace();
                if (!peek('"')) {
                    throw error();
                }
                String key = parseString();
                skipWhitespace();
                if (!peek(':')) {
                    throw error();
                }
                position++;
                result.put(key, parseValue());
                skipWhitespace();
                if (peek('}')) {
                    position++;
                    return result;
                }
                if (!peek(',')) {
                    throw error();
                }
                position++;
            }
        }

        private List<Object> parseArray() {
            List<Object> result = new ArrayList<>();
            position++;
            skipWhitespace();
            if (peek(']')) {
                position++;
                return result;
            }
            while (true) {
                result.add(parseValue());
                skipWhitespace();
                if (peek(']')) {
                    position++;
                    return result;
                }
                if (!peek(',')) {
                    throw error();
                }
                position++;
            }
        }

        private String parseString() {
            position++;
            StringBuilder builder = new StringBuilder();
            while (position < text.length()) {
                char c = text.charAt(position++);
                if (c == '"') {
                    return builder.toString();
                }
                if (c != '\\') {
                    builder.append(c);
                    continue;
                }
                if (position >= text.length()) {
                    throw error();
                }
                char escape = text.charAt(position++);
                switch (escape) {
                    case '"' -> builder.append('"');
                    case '\\' -> builder.append('\\');
                    case '/' -> builder.append('/');
                    case 'b' -> builder.append('\b');
                    case 'f' -> builder.append('\f');
                    case 'n' -> builder.append('\n');
                    case 'r' -> builder.append('\r');
                    case 't' -> builder.append('\t');
                    case 'u' -> {
                        if (position + 4 > text.length()) {
                            throw error();
                        }
                        try {
                            builder.append((char) Integer.parseInt(text.substring(position, position + 4), 16));
                        } catch (NumberFormatException failure) {
                            throw error();
                        }
                        position += 4;
                    }
                    default -> throw error();
                }
            }
            throw error();
        }

        private Object parseNumber() {
            int start = position;
            if (peek('-')) {
                position++;
            }
            while (position < text.length() && isNumberChar(text.charAt(position))) {
                position++;
            }
            if (start == position) {
                throw error();
            }
            try {
                return Double.parseDouble(text.substring(start, position));
            } catch (NumberFormatException failure) {
                throw error();
            }
        }

        private Object parseLiteral(String literal, Object value) {
            if (text.startsWith(literal, position)) {
                position += literal.length();
                return value;
            }
            throw error();
        }

        private static boolean isNumberChar(char c) {
            return (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-';
        }

        private boolean peek(char c) {
            return position < text.length() && text.charAt(position) == c;
        }

        private void skipWhitespace() {
            while (position < text.length() && Character.isWhitespace(text.charAt(position))) {
                position++;
            }
        }

        private InvalidDataException error() {
            return new InvalidDataException("Invalid JSON.");
        }
    }
}
