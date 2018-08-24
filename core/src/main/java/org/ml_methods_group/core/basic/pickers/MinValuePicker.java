package org.ml_methods_group.core.basic.pickers;

import org.ml_methods_group.core.RepresenterPicker;

import java.util.Comparator;
import java.util.List;
import java.util.NoSuchElementException;

public class MinValuePicker<V> implements RepresenterPicker<V> {

    private final Comparator<V> comparator;

    public MinValuePicker(Comparator<V> comparator) {
        this.comparator = comparator;
    }

    @Override
    public V pick(List<V> values) {
        return values.stream()
                .min(comparator)
                .orElseThrow(NoSuchElementException::new);
    }
}
