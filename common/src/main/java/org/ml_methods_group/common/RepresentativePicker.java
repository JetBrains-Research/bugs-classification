package org.ml_methods_group.common;

import java.util.List;

public interface RepresentativePicker<V> {
    V pick(List<V> values);
}
