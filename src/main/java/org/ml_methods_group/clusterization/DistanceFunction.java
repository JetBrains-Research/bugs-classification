package org.ml_methods_group.clusterization;

import java.io.Serializable;

public interface DistanceFunction<T> extends Serializable {
    double distance(T first, T second);
}
