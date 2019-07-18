package org.ml_methods_group.server;

public class Response {
    private String hint;
    private double confidence;
    private String errorMessage;
    private ResponseStatus status;
    private long time;

    private Response(String hint, double confidence, String errorMessage, ResponseStatus status, long time) {
        this.hint = hint;
        this.confidence = confidence;
        this.errorMessage = errorMessage;
        this.status = status;
        this.time = time;
    }

    public String getHint() {
        return hint;
    }

    public void setHint(String hint) {
        this.hint = hint;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public ResponseStatus getStatus() {
        return status;
    }

    public void setStatus(ResponseStatus status) {
        this.status = status;
    }

    public long getTime() {
        return time;
    }

    public void setTime(long time) {
        this.time = time;
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


