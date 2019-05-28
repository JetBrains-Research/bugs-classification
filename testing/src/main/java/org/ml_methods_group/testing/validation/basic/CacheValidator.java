package org.ml_methods_group.testing.validation.basic;

import org.ml_methods_group.common.Repository;
import org.ml_methods_group.common.Database;
import org.ml_methods_group.testing.validation.Validator;

import java.io.*;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.ToIntFunction;

public class CacheValidator<V, M> implements Validator<V, M> {

    private final Function<M, String> encoder;
    private final ToIntFunction<V> idExtractor;
    private final Repository<ValidationData, Boolean> repository;
    private final Validator<V, M> oracle;

    public CacheValidator(Function<M, String> encoder, ToIntFunction<V> idExtractor,
                          Validator<V, M> oracle, Database database) throws Exception {
        this.encoder = encoder;
        this.idExtractor = idExtractor;
        this.repository = database.repositoryForName("validation", ValidationData.class, Boolean.class);
        this.oracle = oracle;
    }

    @Override
    public boolean isValid(V value, M mark) {
        final int valueId = idExtractor.applyAsInt(value);
        final String markCode = encoder.apply(mark);
        final ValidationData validationData = new ValidationData(valueId, markCode);
        final Optional<Boolean> cache = repository.loadValue(validationData);
        if (cache.isPresent()) {
            return cache.get();
        }
        final boolean result = oracle.isValid(value, mark);
        repository.storeValue(validationData, result);
        return result;
    }

    public static class ValidationData implements Externalizable {
        private int valueId;
        private String mark;

        private ValidationData(int valueId, String mark) {
            this.valueId = valueId;
            this.mark = mark;
        }

        @Override
        public void writeExternal(ObjectOutput out) throws IOException {
            out.write(valueId);
            out.writeUTF(mark);
        }

        @Override
        public void readExternal(ObjectInput in) throws IOException {
            valueId = in.readInt();
            mark = in.readUTF();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            ValidationData that = (ValidationData) o;

            if (valueId != that.valueId) return false;
            return Objects.equals(mark, that.mark);
        }

        @Override
        public int hashCode() {
            int result = valueId;
            result = 31 * result + (mark != null ? mark.hashCode() : 0);
            return result;
        }
    }
}
