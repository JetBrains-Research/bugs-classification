package org.ml_methods_group.testing.selectors;

import org.ml_methods_group.common.Database;
import org.ml_methods_group.common.ManyOptionsSelector;
import org.ml_methods_group.common.Repository;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;

public class CacheManyOptionsSelector<V, O> implements ManyOptionsSelector<V, O> {

    private final Repository<Integer, String> repository;

    private final Map<Integer, O> options;
    private final ManyOptionsSelector<V, O> oracle;
    private final ToIntFunction<V> valueIdExtractor;
    private final ToIntFunction<O> optionIdExtractor;
    private final Function<List<Integer>, String> optionsListEncoder;
    private final Function<String, List<Integer>> optionsListDecoder;

    public CacheManyOptionsSelector(ManyOptionsSelector<V, O> oracle, Database database,
                                    ToIntFunction<V> valueIdExtractor,
                                    ToIntFunction<O> optionIdExtractor,
                                    Function<List<Integer>, String> optionsListEncoder,
                                    Function<String, List<Integer>> optionsListDecoder) throws Exception {
        this.options = oracle.getAllPossibleOptions().stream()
                .collect(Collectors.toMap(optionIdExtractor::applyAsInt, Function.identity()));
        long hash = oracle.getAllPossibleOptions().stream()
                .mapToInt(optionIdExtractor)
                .sorted()
                .asLongStream()
                .reduce(0, (h, x) -> h * 37 + x);
        this.repository = database.repositoryForName(
                "option_selector@" + hash + oracle.getSelectionSize(),
                Integer.class, String.class);
        this.oracle = oracle;
        this.valueIdExtractor = valueIdExtractor;
        this.optionIdExtractor = optionIdExtractor;
        this.optionsListEncoder = optionsListEncoder;
        this.optionsListDecoder = optionsListDecoder;
    }

    @Override
    public Collection<O> getAllPossibleOptions() {
        return options.values();
    }

    @Override
    public int getSelectionSize() { return oracle.getSelectionSize(); }

    @Override
    public Optional<List<O>> selectOptions(V value) {
        final int valueId = valueIdExtractor.applyAsInt(value);
        if (valueId < 0) {
            return oracle.selectOptions(value);
        }
        final Optional<List<Integer>> cache = repository.loadValue(valueId).map(optionsListDecoder);
        if (cache.isPresent()) {
            return Optional.of(cache.get().stream()
                    .map(options::get)
                    .collect(Collectors.toList()));
        }
        final Optional<List<O>> options = oracle.selectOptions(value);
        options.ifPresent(list -> {
            var optionsTupleId = optionsListEncoder.apply(list.stream()
                    .map(optionIdExtractor::applyAsInt)
                    .collect(Collectors.toList()));
            repository.storeValue(valueId, optionsTupleId);
        });
        return options;
    }
}
