package org.ml_methods_group.core.basic.validators;

import org.ml_methods_group.core.Validator;
import org.ml_methods_group.core.database.ConditionSupplier;
import org.ml_methods_group.core.database.Database;
import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.database.annotations.DataClass;
import org.ml_methods_group.core.database.annotations.DataField;

import java.util.Optional;
import java.util.function.Function;
import java.util.function.ToIntFunction;

public class CacheValidator<V, M> implements Validator<V, M> {

    private final Function<M, String> encoder;
    private final ToIntFunction<V> idExtractor;
    private final Repository<CachedValidation> repository;
    private final Validator<V, M> oracle;

    public CacheValidator(Function<M, String> encoder, ToIntFunction<V> idExtractor,
                          Validator<V, M> oracle, Database database) {
        this.encoder = encoder;
        this.idExtractor = idExtractor;
        this.repository = database.getRepository(CachedValidation.class);
        this.oracle = oracle;
    }

    @Override
    public boolean isValid(V value, M mark) {
        final int valueId = idExtractor.applyAsInt(value);
        final String markCode = encoder.apply(mark);
        final Optional<Boolean> cache = loadCached(valueId, markCode);
        if (cache.isPresent()) {
            return cache.get();
        }
        final boolean result = oracle.isValid(value, mark);
        storeCached(valueId, markCode, result);
        return result;
    }

    private Optional<Boolean> loadCached(int valueId, String mark) {
        final ConditionSupplier supplier = repository.conditionSupplier();
        return repository.find(supplier.is("valueId", valueId), supplier.is("mark", mark))
                .map(CachedValidation::isAcceptable);
    }

    private void storeCached(int id, String mark, boolean isAcceptable) {
        repository.insert(new CachedValidation(id, mark, isAcceptable));
    }

    @DataClass(defaultStorageName = "validations_cache")
    public static class CachedValidation {
        @DataField
        private final int valueId;
        @DataField
        private final String mark;
        @DataField
        private final boolean isAcceptable;

        private CachedValidation(int valueId, String mark, boolean isAcceptable) {
            this.valueId = valueId;
            this.mark = mark;
            this.isAcceptable = isAcceptable;
        }

        public CachedValidation() {
            this(0, "", false);
        }

        private boolean isAcceptable() {
            return isAcceptable;
        }
    }
}
