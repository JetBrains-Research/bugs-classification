package org.ml_methods_group.common;

import java.io.Serializable;

@FunctionalInterface
public interface DistanceFunction<V> extends Serializable {
    double distance(V first, V second);

    default double distance(V first, V second, double upperBound) {
        return Math.min(distance(first, second), upperBound);
    }

    default double upperBound() {
        return 1;
    }
}
