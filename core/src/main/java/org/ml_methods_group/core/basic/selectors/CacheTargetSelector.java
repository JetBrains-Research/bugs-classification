package org.ml_methods_group.core.basic.selectors;

import org.ml_methods_group.core.TargetSelector;
import org.ml_methods_group.core.database.ConditionSupplier;
import org.ml_methods_group.core.database.Database;
import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.database.annotations.DataClass;
import org.ml_methods_group.core.database.annotations.DataField;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.ToIntFunction;

public class CacheTargetSelector<T> implements TargetSelector<T> {

    private final Repository<CachedDecision> repository;
    private final ConditionSupplier supplier;

    private final Map<Integer, T> targets = new HashMap<>();
    private final TargetSelector<T> oracle;
    private final ToIntFunction<T> idExtractor;

    public CacheTargetSelector(Database database, String identifier,
                               TargetSelector<T> oracle,
                               ToIntFunction<T> idExtractor) {
        this.repository = database.getRepository(identifier, CachedDecision.class);
        this.oracle = oracle;
        this.idExtractor = idExtractor;
        this.supplier = repository.conditionSupplier();
    }

    @Override
    public void addTarget(T target) {
        targets.put(idExtractor.applyAsInt(target), target);
        oracle.addTarget(target);
    }

    @Override
    public T selectTarget(T value) {
        final int id = idExtractor.applyAsInt(value);
        final Optional<T> cache = loadCached(id).map(targets::get);
        if (cache.isPresent()) {
            return cache.get();
        }
        final T target = oracle.selectTarget(value);
        storeCached(id, idExtractor.applyAsInt(target));
        return target;
    }

    private Optional<Integer> loadCached(int id) {
        return repository.find(supplier.is("valueid", id)).map(CachedDecision::getTargetId);
    }

    private void storeCached(int id, int targetId) {
        repository.insert(new CachedDecision(id, targetId));
    }

    @DataClass(defaultStorageName = "decisions_cache")
    public static class CachedDecision {
        @DataField
        private final int valueId;
        @DataField
        private final int targetId;

        public CachedDecision() {
            this(0, 0);
        }

        public CachedDecision(int valueId, int targetId) {
            this.valueId = valueId;
            this.targetId = targetId;
        }

        public int getValueId() {
            return valueId;
        }

        public int getTargetId() {
            return targetId;
        }
    }
}
