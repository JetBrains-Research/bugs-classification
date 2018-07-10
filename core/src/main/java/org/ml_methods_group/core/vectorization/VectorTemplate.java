package org.ml_methods_group.core.vectorization;

import org.ml_methods_group.core.changes.AtomicChange;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class VectorTemplate implements Serializable {
    private final List<EncodingStrategy> strategies;
    private final Map<Long, Integer> codeToIndex = new HashMap<>();
    private final Postprocessor postprocessor;

    public VectorTemplate(List<Long> acceptable, Postprocessor postprocessor,
                          EncodingStrategy... strategies) {
        this(acceptable, postprocessor, Arrays.asList(strategies));
    }

    public VectorTemplate(List<Long> acceptable, Postprocessor postprocessor,
                          List<EncodingStrategy> strategies) {
        this.postprocessor = postprocessor;
        this.strategies = strategies;
        acceptable.forEach(code -> codeToIndex.putIfAbsent(code, codeToIndex.size()));
    }

    public double[] process(List<AtomicChange> features) {
        final double[] result = new double[codeToIndex.size()];
        final HashMap<Long, Integer> buffer = new HashMap<>();
        for (EncodingStrategy strategy : strategies) {
            for (AtomicChange feature : features) {
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
                final double norma = MathUtils.norm(vector);
                for (int i = 0; i < vector.length; i++) {
                    vector[i] /= norma;
                }
            }
        };
    }
}
