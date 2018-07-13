package org.ml_methods_group.core.parallel;

import java.util.List;
import java.util.Map;

public class ParallelUtils {
    public static <V> List<V> combineLists(List<V> first, List<V> second) {
        if (first.size() < second.size()) {
            return combineLists(second, first);
        }
        first.addAll(second);
        return first;
    }

    public static <K, V> Map<K, V> combineMaps(Map<K, V> first, Map<K, V> second) {
        if (first.size() < second.size()) {
            return combineMaps(second, first);
        }
        first.putAll(second);
        return first;
    }

    public static <K> Map<K, Integer> combineCounters(Map<K, Integer> first, Map<K, Integer> second) {
        if (first.size() < second.size()) {
            return combineMaps(second, first);
        }
        for (Map.Entry<K, Integer> entry : second.entrySet()) {
            first.compute(entry.getKey(), (key, old) -> (old == null ? 0 : old) + entry.getValue());
        }
        return first;
    }
}
