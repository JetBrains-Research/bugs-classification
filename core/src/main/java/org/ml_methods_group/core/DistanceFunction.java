package org.ml_methods_group.core;

import java.io.Serializable;

@FunctionalInterface
public interface DistanceFunction<V> extends Serializable {
    double distance(V first, V second);

    default double distance(Wrapper<V> first, Wrapper<V> second) {
        return distance(first.getFeatures(), second.getFeatures());
    }

    default double distance(V first, V second, double upperBound) {
        return Math.min(distance(first, second), upperBound);
    }

    default double distance(Wrapper<V> first, Wrapper<V> second, double upperBound) {
        return distance(first.getFeatures(), second.getFeatures(), upperBound);
    }
}
