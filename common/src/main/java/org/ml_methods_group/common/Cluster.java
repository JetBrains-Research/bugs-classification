package org.ml_methods_group.common;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Cluster<V> implements Iterable<V>, Serializable {
    private final List<V> elements;

    public Cluster(List<V> elements) {
        this.elements = new ArrayList<>(elements);
    }

    public int size() {
        return elements.size();
    }

    public <T> Cluster<T> map(Function<? super V, T> mapping) {
        final List<T> buffer = elements.stream()
                .map(mapping)
                .collect(Collectors.toList());
        return new Cluster<>(buffer);
    }

    public Cluster<V> merge(Cluster<V> other) {
        final List<V> buffer = new ArrayList<>(elements.size() + other.elements.size());
        buffer.addAll(elements);
        buffer.addAll(other.elements);
        return new Cluster<>(buffer);
    }

    public List<V> getElements() {
        return Collections.unmodifiableList(elements);
    }

    public List<V> elementsCopy() {
        return new ArrayList<>(elements);
    }

    @Override
    public Iterator<V> iterator() {
        return elements.iterator();
    }

    public Stream<V> stream() {
        return elements.stream();
    }
}
