package org.ml_methods_group.common;

import java.util.List;

public interface ManyRepresentativesPicker<V> {
    List<V> pick(List<V> values);
    int getSelectionSize();
}
