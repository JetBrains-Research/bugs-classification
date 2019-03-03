package org.ml_methods_group.common;

import java.io.Serializable;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class Dataset<T> implements Serializable, Iterable<T> {
    private final List<T> values;

    public Dataset(Collection<T> values) {
        this(new ArrayList<>(values));
    }

    private Dataset(ArrayList<T> values) {
        this.values = values;
    }

    public Dataset<T> filter(Predicate<T> predicate) {
        return new Dataset<T>(getValues(predicate));
    }

    public List<T> getValues() {
        return Collections.unmodifiableList(values);
    }

    public List<T> getValues(Predicate<T> predicate) {
        return values.stream()
                .filter(predicate)
                .collect(Collectors.toList());
    }

    @Override
    public Iterator<T> iterator() {
        return values.iterator();
    }
}
