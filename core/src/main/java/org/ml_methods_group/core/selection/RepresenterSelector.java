package org.ml_methods_group.core.selection;

import java.util.List;

public interface RepresenterSelector<T> {
    List<T> findRepresenter(int n, List<T> samples);
}
