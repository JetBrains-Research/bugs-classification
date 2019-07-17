package org.ml_methods_group.server;

public class Response {
    private final String hint;
    private final double confidence;
    private final String errorMessage;
    private final ResponseStatus status;
    private final long time;

    Response(String hint, double confidence, String errorMessage, ResponseStatus status, long time) {
        this.hint = hint;
        this.confidence = confidence;
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

    static Response success(String hint, double confidence, long requestTime) {
        return new Response(hint, confidence,"", ResponseStatus.OK,
                System.currentTimeMillis() - requestTime);
    }

    static Response error(String message, long requestTime) {
        return new Response("", 0, message, ResponseStatus.ERROR,
                System.currentTimeMillis() - requestTime);
    }
}


