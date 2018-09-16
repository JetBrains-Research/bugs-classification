package org.ml_methods_group.core;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class MarkedClusters<V, M> implements Serializable {
    private final Map<Cluster<V>, M> marks;

    public MarkedClusters(Map<Cluster<V>, M> marks) {
        this(new HashMap<>(marks));
    }

    private MarkedClusters(HashMap<Cluster<V>, M> marks) {
        this.marks = new HashMap<>(marks);
    }

    public Map<Cluster<V>, M> getMarks() {
        return Collections.unmodifiableMap(marks);
    }

    public Map<V, M> getFlatMarks() {
        final HashMap<V, M> buffer = new HashMap<>();
        for (Map.Entry<Cluster<V>, M> entry : marks.entrySet()) {
            final M mark = entry.getValue();
            entry.getKey().forEach(x -> buffer.put(x, mark));
        }
        return buffer;
    }

    public <T> MarkedClusters<T, M> map(Function<V, T> mapping) {
        final HashMap<Cluster<T>, M> buffer = marks.entrySet()
                .stream()
                .collect(Collectors.toMap(
                        entry -> entry.getKey().map(mapping),
                        Map.Entry::getValue,
                        (M a, M b) -> {
                            throw new RuntimeException("Unexpected situation");
                        },
                        HashMap::new));
        return new MarkedClusters<>(buffer);
    }
}
