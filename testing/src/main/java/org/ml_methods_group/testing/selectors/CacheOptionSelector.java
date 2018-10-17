package org.ml_methods_group.testing.selectors;

import org.ml_methods_group.common.OptionSelector;
import org.ml_methods_group.testing.database.ConditionSupplier;
import org.ml_methods_group.testing.database.Database;
import org.ml_methods_group.testing.database.Repository;

import java.util.*;
import java.util.function.Function;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;

public class CacheOptionSelector<V, O> implements OptionSelector<V, O> {

    private final Repository<CachedDecision> repository;
    private final ConditionSupplier supplier;

    private final Map<Integer, O> options;
    private final OptionSelector<V, O> oracle;
    private final ToIntFunction<V> valueIdExtractor;
    private final ToIntFunction<O> optionIdExtractor;
    private final long hash;

    public CacheOptionSelector(OptionSelector<V, O> oracle, Database database,
                               ToIntFunction<V> valueIdExtractor, ToIntFunction<O> optionIdExtractor) {
        this.repository = database.getRepository(CachedDecision.class);
        this.supplier = repository.conditionSupplier();
        this.oracle = oracle;
        this.valueIdExtractor = valueIdExtractor;
        this.optionIdExtractor = optionIdExtractor;
        final Collection<O> options = oracle.getOptions();
        this.options = options.stream()
                .collect(Collectors.toMap(optionIdExtractor::applyAsInt, Function.identity()));
        hash = options.stream()
                .mapToInt(optionIdExtractor)
                .sorted()
                .asLongStream()
                .reduce(0, (h, x) -> h * 37 + x);
    }

    @Override
    public Collection<O> getOptions() {
        return options.values();
    }

    @Override
    public Optional<O> selectOption(V value) {
        final int valueId = valueIdExtractor.applyAsInt(value);
        final Optional<O> cache = loadCached(valueId).map(options::get);
        if (cache.isPresent()) {
            return cache;
        }
        final Optional<O> option = oracle.selectOption(value);
        option.map(optionIdExtractor::applyAsInt)
                .ifPresent(id -> storeCached(valueId, id));
        return option;
    }

    private Optional<Integer> loadCached(int valueId) {
        return repository.find(
                supplier.is("valueId", valueId),
                supplier.is("hash", hash))
                .map(CachedDecision::getOptionId);
    }

    private void storeCached(int id, int targetId) {
        repository.insert(new CachedDecision(id, targetId, hash));
    }

    @SuppressWarnings("unused")
    public static class CachedDecision {
        private final int valueId;
        private final int optionId;
        private final long hash;

        public CachedDecision() {
            this(0, 0, 0);
        }

        private CachedDecision(int valueId, int optionId, long hash) {
            this.valueId = valueId;
            this.optionId = optionId;
            this.hash = hash;
        }

        private int getOptionId() {
            return optionId;
        }
    }
}
