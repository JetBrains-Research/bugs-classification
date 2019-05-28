package org.ml_methods_group.common;

import javax.annotation.Nonnull;
import java.io.Serializable;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class Dataset implements Serializable, Iterable<Solution> {
    private final List<Solution> values;

    public Dataset(Collection<Solution> values) {
        this(new ArrayList<>(values));
    }

    private Dataset(ArrayList<Solution> values) {
        this.values = values;
    }

    public Dataset filter(Predicate<Solution> predicate) {
        return new Dataset(getValues(predicate));
    }

    public List<Solution> getValues() {
        return Collections.unmodifiableList(values);
    }

    public List<Solution> getValues(Predicate<Solution> predicate) {
        return values.stream()
                .filter(predicate)
                .collect(Collectors.toList());
    }

    @Nonnull
    @Override
    public Iterator<Solution> iterator() {
        return values.iterator();
    }
}
