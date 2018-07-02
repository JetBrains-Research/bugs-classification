package org.ml_methods_group.vectorization.index;

import org.ml_methods_group.core.IndexDatabase;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.vectorization.EncodingStrategy;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DiffIndexer {
    public static void indexDiffs(SolutionDatabase solutions,
                                  List<EncodingStrategy<AtomicChange>> strategies,
                                  IndexDatabase storage, String indexName) {
        final Map<Long, Long> index = new HashMap<>();
        for (EncodingStrategy<AtomicChange> strategy : strategies) {
            final Map<Long, Long> buffer = solutions.getDiffs().stream()
                    .map(SolutionDiff::getChanges)
                    .flatMap(List::stream)
                    .map(strategy::encode)
                    .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
            for (Entry<Long, Long> entry : buffer.entrySet()) {
                final Long old = index.putIfAbsent(entry.getKey(), entry.getValue());
                if (old != null && !old.equals(entry.getValue())) {
                    throw new RuntimeException("Conflict encoding!");
                }
            }
        }
    }

    public static Map<Long, Long> getDiffIndex(IndexDatabase storage, String indexName) {
        return storage.loadIndex(indexName, Long::parseLong, Long::parseLong);
    }
}
