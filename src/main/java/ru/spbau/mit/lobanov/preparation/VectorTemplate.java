package ru.spbau.mit.lobanov.preparation;

import ru.spbau.mit.lobanov.changes.AtomicChange;
import ru.spbau.mit.lobanov.changes.EncodingStrategy;

import java.util.*;

public class VectorTemplate {
    private final List<Long> featuresTypes;
    private final Map<Long, Integer> typeToIndex;
    private final EncodingStrategy[] strategies;

    public VectorTemplate(Collection<Long> featuresTypes, EncodingStrategy... strategies) {
        this.featuresTypes = new ArrayList<>(featuresTypes);
        this.typeToIndex = new HashMap<>();
        this.strategies = strategies;
        for (int i = 0; i < featuresTypes.size(); i++) {
            typeToIndex.put(this.featuresTypes.get(i), i);
        }
    }

    public int getIndex(long type) {
        return typeToIndex.getOrDefault(type, -1);
    }

    public long getType(int index) {
        return featuresTypes.get(index);
    }

    public int size() {
        return featuresTypes.size();
    }

    public double[] toVector(List<AtomicChange> changes) {
        final double[] result = new double[featuresTypes.size()];
        final HashMap<Long, Integer> buffer = new HashMap<>();
        for (EncodingStrategy strategy : strategies) {
            for (AtomicChange change : changes) {
                PreparationUtils.incrementCounter(buffer, strategy.encode(change));
            }
            for (Map.Entry<Long, Integer> counter : buffer.entrySet()) {
                final int index = getIndex(counter.getKey());
                final int count = counter.getValue();
                if (index == -1 || result[index] == count) {
                    continue;
                } else if (result[index] != 0) {
                    throw new RuntimeException("Encoding conflict detected!");
                }
                result[index] = count;
            }
            buffer.clear();
        }
        for (int i = 0; i < result.length; i++) {
            result[i] /= changes.size();
        }
        return result;
    }
}
