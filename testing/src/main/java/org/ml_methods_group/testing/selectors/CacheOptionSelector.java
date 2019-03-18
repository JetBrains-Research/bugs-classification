package org.ml_methods_group.testing.selectors;

import org.ml_methods_group.common.OptionSelector;
import org.ml_methods_group.common.Repository;
import org.ml_methods_group.common.Database;

import java.util.*;
import java.util.function.Function;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;

public class CacheOptionSelector<V, O> implements OptionSelector<V, O> {

    private final Repository<Integer, Integer> repository;

    private final Map<Integer, O> options;
    private final OptionSelector<V, O> oracle;
    private final ToIntFunction<V> valueIdExtractor;
    private final ToIntFunction<O> optionIdExtractor;

    public CacheOptionSelector(OptionSelector<V, O> oracle, Database database,
                               ToIntFunction<V> valueIdExtractor, ToIntFunction<O> optionIdExtractor) throws Exception {
        this.options = oracle.getOptions().stream()
                .collect(Collectors.toMap(optionIdExtractor::applyAsInt, Function.identity()));
        long hash = oracle.getOptions().stream()
                .mapToInt(optionIdExtractor)
                .sorted()
                .asLongStream()
                .reduce(0, (h, x) -> h * 37 + x);
        this.repository = database.repositoryForName("option_selector@" + hash, Integer.class, Integer.class);
        this.oracle = oracle;
        this.valueIdExtractor = valueIdExtractor;
        this.optionIdExtractor = optionIdExtractor;

    }

    @Override
    public Collection<O> getOptions() {
        return options.values();
    }

    @Override
    public Optional<O> selectOption(V value) {
        final int valueId = valueIdExtractor.applyAsInt(value);
        if (valueId < 0) {
            return oracle.selectOption(value);
        }
        final Optional<O> cache = repository.loadValue(valueId).map(options::get);
        if (cache.isPresent()) {
            return cache;
        }
        final Optional<O> option = oracle.selectOption(value);
        option.map(optionIdExtractor::applyAsInt)
                .ifPresent(id -> repository.storeValue(valueId, id));
        return option;
    }
}
