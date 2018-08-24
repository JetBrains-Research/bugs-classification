package org.ml_methods_group.core;

import java.util.List;

public interface RepresenterPicker<V> {
    V pick(List<V> values);
}
