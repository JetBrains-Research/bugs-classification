package org.ml_methods_group.common.preparation;

import java.util.List;

public interface ValuePicker<V> {
    V pick(List<V> values);
}
