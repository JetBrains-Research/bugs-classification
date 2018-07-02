package org.ml_methods_group.core;

import java.io.Serializable;

public interface DistanceFunction<V> extends Serializable {
    double distance(V first, V second);
}
