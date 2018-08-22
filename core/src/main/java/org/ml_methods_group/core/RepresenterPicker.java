package org.ml_methods_group.core;

import java.util.List;

public interface RepresenterSelector<T> {
    T getCenter(List<T> cluster);
}
