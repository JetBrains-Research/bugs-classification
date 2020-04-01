package org.ml_methods_group.common.preparation.basic;

import org.ml_methods_group.common.preparation.ValuePicker;
import org.ml_methods_group.common.preparation.Unifier;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;

public class BasicUnifier<V> implements Unifier<V> {
    private final ToIntFunction<V> hashFunction;
    private final BiPredicate<V, V> equalsFunction;
    private final ValuePicker<V> picker;

    public BasicUnifier(ToIntFunction<V> hashFunction,
                        BiPredicate<V, V> equalsFunction,
                        ValuePicker<V> picker) {
        this.hashFunction = hashFunction;
        this.equalsFunction = equalsFunction;
        this.picker = picker;
    }

    @Override
    public List<V> unify(List<V> values) {
        return values.stream()
                .map(Holder::new)
                .collect(Collectors.groupingBy(Function.identity()))
                .values()
                .stream()
                .map(this::holdersToValues)
                .map(picker::pick)
                .collect(Collectors.toList());
    }

    private class Holder {
        private final V value;
        private final int hash;

        private Holder(V value) {
            this.value = value;
            this.hash = hashFunction.applyAsInt(value);
        }

        @Override
        @SuppressWarnings("unchecked")
        public boolean equals(Object other) {
            if (other == null || other.getClass() != Holder.class) {
                return false;
            }
            final Holder holder = (Holder) other;
            return hash == holder.hash && equalsFunction.test(value, holder.value);
        }

        @Override
        public int hashCode() {
            return hash;
        }

        public V getValue() {
            return value;
        }
    }

    private List<V> holdersToValues(List<Holder> holders) {
        final List<V> buffer = new ArrayList<>(holders.size());
        holders.forEach(x -> buffer.add(x.value));
        return buffer;
    }
}
