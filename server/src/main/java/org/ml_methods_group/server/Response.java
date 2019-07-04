package org.ml_methods_group.server;

import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement
public class Response {
    private String hint;
    private String errorMessage;
    private double confidence;
    private ResponseStatus status;

    private Response(String hint, String errorMessage, double confidence, ResponseStatus status) {
        this.hint = hint;
        this.errorMessage = errorMessage;
        this.confidence = confidence;
        this.status = status;
    }

    public enum ResponseStatus {OK, ERROR}

    static Response success(String hint, double confidence) {
        return new Response(hint, "", confidence, ResponseStatus.OK);
    }

    static Response error(String message) {
        return new Response("", message, 0, ResponseStatus.ERROR);
    }
}


