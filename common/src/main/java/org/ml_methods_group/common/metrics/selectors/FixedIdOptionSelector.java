package org.ml_methods_group.common.metrics.selectors;

import org.ml_methods_group.core.OptionSelector;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;

public class FixedIdOptionSelector<V, K, O> implements OptionSelector<V, O> {

    private final Map<K, O> options;
    private final Function<V, K> valueIdExtractor;

    public FixedIdOptionSelector(List<O> options, Function<V, K> valueIdExtractor, Function<O, K> optionIdExtractor) {
        this.valueIdExtractor = valueIdExtractor;
        this.options = options.stream()
                .collect(Collectors.toMap(optionIdExtractor, Function.identity()));
    }

    @Override
    public Optional<O> selectOption(V value) {
        return Optional.ofNullable(options.get(valueIdExtractor.apply(value)));
    }

    @Override
    public Collection<O> getOptions() {
        return options.values();
    }
}
