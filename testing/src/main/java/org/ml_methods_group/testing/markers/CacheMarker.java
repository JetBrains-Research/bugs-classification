package org.ml_methods_group.testing.markers;

import org.ml_methods_group.markers.Marker;
import org.ml_methods_group.testing.database.ConditionSupplier;
import org.ml_methods_group.testing.database.Database;
import org.ml_methods_group.testing.database.Repository;
import org.ml_methods_group.testing.database.annotations.DataClass;
import org.ml_methods_group.testing.database.annotations.DataField;

import java.util.Optional;
import java.util.function.Function;
import java.util.function.ToIntFunction;

public class CacheMarker<V, M> implements Marker<V, M> {

    private final Repository<CachedMark> repository;
    private final ConditionSupplier supplier;
    private final Function<M, String> printer;
    private final Function<String, M> parser;
    private final ToIntFunction<V> idExtractor;
    private final Marker<V, M> oracle;

    public CacheMarker(Function<M, String> printer, Function<String, M> parser,
                       ToIntFunction<V> idExtractor, Database database, Marker<V, M> oracle) {
        this.repository = database.getRepository(CachedMark.class);
        this.supplier = repository.conditionSupplier();
        this.printer = printer;
        this.parser = parser;
        this.idExtractor = idExtractor;
        this.oracle = oracle;
    }

    @Override
    public M mark(V value) {
        final int valueId = idExtractor.applyAsInt(value);
        final Optional<M> cache = loadCached(valueId);
        if (cache.isPresent()) {
            return cache.get();
        }
        final M mark = oracle.mark(value);
        if (mark != null) {
            storeCached(valueId, mark);
        }
        return mark;
    }

    public void cacheMark(V value, M mark) {
        final int valueId = idExtractor.applyAsInt(value);
        if (loadCached(valueId).isPresent()) {
            //todo
            return;
        }
        storeCached(valueId, mark);
    }

    private Optional<M> loadCached(int valueId) {
        return repository.find(supplier.is("valueId", valueId))
                .map(CachedMark::getMark)
                .map(parser);
    }

    private void storeCached(int id, M mark) {
        repository.insert(new CachedMark(id, printer.apply(mark)));
    }

    @DataClass(defaultStorageName = "marks_cache")
    public static class CachedMark {
        @DataField
        private final int valueId;
        @DataField
        private final String mark;

        private CachedMark(int valueId, String mark) {
            this.valueId = valueId;
            this.mark = mark;
        }

        public CachedMark() {
            this(0, "");
        }

        private String getMark() {
            return mark;
        }
    }
}
