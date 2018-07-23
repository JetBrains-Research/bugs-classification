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

    public static <T> DistanceFunction<Wrapper<T>> wrap(DistanceFunction<T> metric) {
        return new DistanceFunction<Wrapper<T>>() {
            @Override
            public double distance(Wrapper<T> first, Wrapper<T> second) {
                return metric.distance(first.getFeatures(), second.getFeatures());
            }

            @Override
            public double distance(Wrapper<T> first, Wrapper<T> second, double upperBound) {
                return metric.distance(first.getFeatures(), second.getFeatures(), upperBound);
            }
        };
    }

    @Override
    public String toString() {
        return "Wrapper{session=" + sessionId + "}";
    }
}
