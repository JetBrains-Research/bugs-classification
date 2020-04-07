package org.ml_methods_group.common;

import java.util.Collection;
import java.util.HashSet;
import java.util.Optional;

public interface OptionSelector<V, O> {
    Optional<O> selectOption(V value);
    Collection<O> getOptions();

    default OptionSelector<V, O> or(OptionSelector<V, O> other) {
        return new OptionSelector<V, O>() {
            @Override
            public Optional<O> selectOption(V value) {
                return OptionSelector.this.selectOption(value)
                        .map(Optional::of)
                        .orElseGet(() -> other.selectOption(value));
            }

            @Override
            public Collection<O> getOptions() {
                final HashSet<O> options = new HashSet<>();
                options.addAll(OptionSelector.this.getOptions());
                options.addAll(other.getOptions());
                return options;
            }
        };
    }
}
