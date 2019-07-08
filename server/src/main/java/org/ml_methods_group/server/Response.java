package org.ml_methods_group.server;

import java.util.Collections;
import java.util.Map;

public class Response {
    private final String hint;
    private final double confidence;
    private final Map<String, Double> hints;
    private final String errorMessage;
    private final ResponseStatus status;
    private final long time;

    Response(String hint, double confidence, Map<String, Double> hints,
             String errorMessage, ResponseStatus status, long time) {
        this.hint = hint;
        this.confidence = confidence;
        this.hints = hints;
        this.errorMessage = errorMessage;
        this.status = status;
        this.time = time;
    }

    public String getHint() {
        return hint;
    }

    public double getConfidence() {
        return confidence;
    }

    public Map<String, Double> getHints() {
        return hints;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public ResponseStatus getStatus() {
        return status;
    }

    public long getTime() {
        return time;
    }

    public enum ResponseStatus {OK, ERROR}

    static Response success(Map<String, Double> hints, long requestTime) {
        final var hint = hints.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .orElseGet(() -> Map.entry("", 0.0));
        return new Response(hint.getKey(), hint.getValue(), hints, "", ResponseStatus.OK,
                System.currentTimeMillis() - requestTime);
    }

    static Response error(String message, long requestTime) {
        return new Response("", 0, Collections.emptyMap(), message, ResponseStatus.ERROR,
                System.currentTimeMillis() - requestTime);
    }
}


