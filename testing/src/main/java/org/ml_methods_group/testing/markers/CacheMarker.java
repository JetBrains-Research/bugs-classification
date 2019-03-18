package org.ml_methods_group.testing.markers;

import org.ml_methods_group.common.Repository;
import org.ml_methods_group.common.Database;
import org.ml_methods_group.marking.markers.Marker;

import java.util.Optional;
import java.util.function.ToIntFunction;

public class CacheMarker<V, M> implements Marker<V, M> {

    private final Repository<Integer, M> repository;
    private final ToIntFunction<V> idExtractor;
    private final Marker<V, M> oracle;

    public CacheMarker(ToIntFunction<V> idExtractor, Class<M> markClass, Marker<V, M> oracle,
                       Database database) throws Exception {
        this.repository = database.repositoryForName("marks_cache", Integer.class, markClass);
        this.idExtractor = idExtractor;
        this.oracle = oracle;
    }

    public void cacheMark(V value, M mark) {
        repository.storeValue(idExtractor.applyAsInt(value), mark);
    }

    @Override
    public M mark(V value) {
        final int valueId = idExtractor.applyAsInt(value);
        final Optional<M> cache = repository.loadValue(valueId);
        if (cache.isPresent()) {
            return cache.get();
        }
        final M mark = oracle.mark(value);
        if (mark != null) {
            repository.storeValue(valueId, mark);
        }
        return mark;
    }
}
