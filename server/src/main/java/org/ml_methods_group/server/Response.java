package org.ml_methods_group.server;

import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement(name = "Response")
public class Response {
    private final String hint;
    private final String errorMessage;
    private final double confidence;
    private final ResponseStatus status;

    Response(String hint, String errorMessage, double confidence, ResponseStatus status) {
        this.hint = hint;
        this.errorMessage = errorMessage;
        this.confidence = confidence;
        this.status = status;
    }

    public String getHint() {
        return hint;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public double getConfidence() {
        return confidence;
    }

    public ResponseStatus getStatus() {
        return status;
    }

    @XmlRootElement(name = "Status")
    public enum ResponseStatus {OK, ERROR}

    static Response success(String hint, double confidence) {
        return new Response(hint, "", confidence, ResponseStatus.OK);
    }

    static Response error(String message) {
        return new Response("", message, 0, ResponseStatus.ERROR);
    }
}


