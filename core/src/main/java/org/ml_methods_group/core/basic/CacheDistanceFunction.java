package org.ml_methods_group.core.basic;

import org.ml_methods_group.core.DistanceFunction;
import org.ml_methods_group.core.database.ConditionSupplier;
import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.CachedDistance;

import java.util.Optional;
import java.util.function.ToIntFunction;

public class CacheDistanceFunction<T> implements DistanceFunction<T> {
    private final Repository<CachedDistance> repository;
    private final ConditionSupplier supplier;
    private final ToIntFunction<T> idExtractor;
    private final DistanceFunction<T> oracle;

    public CacheDistanceFunction(Repository<CachedDistance> repository, ToIntFunction<T> idExtractor,
                                 DistanceFunction<T> oracle) {
        this.repository = repository;
        this.supplier = repository.conditionSupplier();
        this.idExtractor = idExtractor;
        this.oracle = oracle;
    }

    @Override
    public double distance(T first, T second) {
        final int firstId = idExtractor.applyAsInt(first);
        final int secondId = idExtractor.applyAsInt(second);
        final Optional<Double> cache = loadCached(firstId, secondId);
        if (cache.isPresent()) {
            return cache.get();
        }
        final double result = oracle.distance(first, second);
        storeCached(firstId, secondId, result);
        return result;
    }

    private Optional<Double> loadCached(int firstId, int secondId) {
        return repository.find(supplier.is("firstid", firstId), supplier.is("secondid", secondId))
                .map(CachedDistance::getDistance);
    }

    private void storeCached(int firstId, int secondId, double value) {
        repository.insert(new CachedDistance(firstId, secondId, value));
    }
}
