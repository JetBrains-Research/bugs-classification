package org.ml_methods_group.common;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

public interface ManyOptionsSelector<V,O> {
    Optional<List<O>> selectOptions(V value);
    Collection<O> getAllPossibleOptions();
    int getSelectionSize();
}
