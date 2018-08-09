package org.ml_methods_group.core;

public class Wrapper<F, M> {
    private final F features;
    private final M meta;

    public Wrapper(F features, M meta) {
        this.features = features;
        this.meta = meta;
    }

    public F getFeatures() {
        return features;
    }

    public M getMeta() {
        return meta;
    }

    public static <T, M> DistanceFunction<Wrapper<T, M>> wrap(DistanceFunction<T> metric) {
        return new DistanceFunction<Wrapper<T, M>>() {
            @Override
            public double distance(Wrapper<T, M> first, Wrapper<T, M> second) {
                return metric.distance(first.getFeatures(), second.getFeatures());
            }

            @Override
            public double distance(Wrapper<T, M> first, Wrapper<T, M> second, double upperBound) {
                return metric.distance(first.getFeatures(), second.getFeatures(), upperBound);
            }
        };
    }

    @Override
    public String toString() {
        return "Wrapper{meta=" + meta + "}";
    }
}
