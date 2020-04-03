package org.ml_methods_group.testing.representatives;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.Database;
import org.ml_methods_group.common.Repository;
import org.ml_methods_group.common.RepresentativesPicker;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;

public class CacheRepresentativesPicker<V> implements RepresentativesPicker<V> {

    private final Repository<String, String> repository;
    private final Map<Integer, V> options;
    private final RepresentativesPicker<V> oracle;
    private final ToIntFunction<V> valueIdExtractor;
    private final ToIntFunction<V> optionIdExtractor;
    private final Function<List<Integer>, String> listEncoder;
    private final Function<String, List<Integer>> listDecoder;

    public CacheRepresentativesPicker(RepresentativesPicker<V> oracle,
                                      List<V> options,
                                      Database database,
                                      ToIntFunction<V> valueIdExtractor,
                                      ToIntFunction<V> optionIdExtractor,
                                      Function<List<Integer>, String> listEncoder,
                                      Function<String, List<Integer>> listDecoder) throws Exception {
        this.options = options.stream()
                .collect(Collectors.toMap(optionIdExtractor::applyAsInt, Function.identity()));
        long hash = options.stream()
                .mapToInt(optionIdExtractor)
                .sorted()
                .asLongStream()
                .reduce(0, (h, x) -> h * 37 + x);
        this.repository = database.repositoryForName("representatives_picker@" + hash, String.class, String.class);
        this.oracle = oracle;
        this.valueIdExtractor = valueIdExtractor;
        this.optionIdExtractor = optionIdExtractor;
        this.listEncoder = listEncoder;
        this.listDecoder = listDecoder;
    }

    @Override
    public List<V> getRepresentatives(Cluster<V> values) {
        String valueId = listEncoder.apply(values.getElements().stream()
                .map(valueIdExtractor::applyAsInt)
                .collect(Collectors.toList()));
        final Optional<List<Integer>> cache = repository.loadValue(valueId).map(listDecoder);
        if (cache.isPresent()) {
            return cache.get().stream()
                    .map(options::get)
                    .collect(Collectors.toList());
        }
        final List<V> representatives = oracle.getRepresentatives(values);
        var representativesId = listEncoder.apply(representatives.stream()
                .map(optionIdExtractor::applyAsInt)
                .collect(Collectors.toList()));
        repository.storeValue(valueId, representativesId);
        return representatives;
    }
}
