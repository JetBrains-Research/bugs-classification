package org.ml_methods_group.core;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class MarkedClusters<V, M> implements Serializable {
    private final Map<Cluster<V>, M> marks;

    public MarkedClusters(Map<Cluster<V>, M> marks) {
        this.marks = new HashMap<>(marks);
    }

    public Map<Cluster<V>, M> getClusters() {
        return Collections.unmodifiableMap(marks);
    }
}
