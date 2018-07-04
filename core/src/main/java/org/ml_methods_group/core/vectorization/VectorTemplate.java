package org.ml_methods_group.core.vectorization;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class VectorTemplate<T> implements Serializable {
    private final List<EncodingStrategy<T>> strategies;
    private final Map<Long, Integer> codeToIndex = new HashMap<>();
    private final Postprocessor postprocessor;

    @SafeVarargs
    public VectorTemplate(List<Long> acceptable, Postprocessor postprocessor,
                          EncodingStrategy<T>... strategies) {
        this(acceptable, postprocessor, Arrays.asList(strategies));
    }

    public VectorTemplate(List<Long> acceptable, Postprocessor postprocessor,
                          List<EncodingStrategy<T>> strategies) {
        this.postprocessor = postprocessor;
        this.strategies = strategies;
        acceptable.forEach(code -> codeToIndex.putIfAbsent(code, codeToIndex.size()));
    }

    public double[] process(List<T> features) {
        final double[] result = new double[codeToIndex.size()];
        final HashMap<Long, Integer> buffer = new HashMap<>();
        for (EncodingStrategy<? super T> strategy : strategies) {
            for (T feature : features) {
                final long code = strategy.encode(feature);
                if (code != 0) {
                    buffer.compute(code, (key, old) -> (old == null ? 0 : old) + 1);
                }
            }
            for (Map.Entry<Long, Integer> counter : buffer.entrySet()) {
                final int index = codeToIndex.getOrDefault(counter.getKey(), -1);
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
        postprocessor.process(features, result);
        return result;
    }



    public interface Postprocessor extends Serializable {
        void process(List<?> features, double[] vector);
    }

    public enum BasePostprocessors implements Postprocessor {
        NONE {
            @Override
            public void process(List<?> features, double[] vector) {
            }
        },
        RELATIVE {
            @Override
            public void process(List<?> features, double[] vector) {
                for (int i = 0; i < vector.length; i++) {
                    vector[i] /= features.size();
                }
            }
        },
        STANDARD {
            @Override
            public void process(List<?> features, double[] vector) {
                final double norma = Utils.norm(vector);
                for (int i = 0; i < vector.length; i++) {
                    vector[i] /= norma;
                }
            }
        };
    }
}
