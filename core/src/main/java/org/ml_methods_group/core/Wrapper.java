package org.ml_methods_group.core;

public class Wrapper<F> {
    private final F features;
    private final int sessionId;

    public Wrapper(F features, int sessionId) {
        this.features = features;
        this.sessionId = sessionId;
    }

    public F getFeatures() {
        return features;
    }

    public int getSessionId() {
        return sessionId;
    }
}
