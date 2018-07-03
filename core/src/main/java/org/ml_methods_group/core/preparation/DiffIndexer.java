package org.ml_methods_group.core.preparation;

import org.ml_methods_group.core.IndexStorage;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.Utils;
import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.vectorization.EncodingStrategy;

import java.util.*;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DiffIndexer {

    private final SolutionDatabase database;
    private final IndexStorage storage;

    public DiffIndexer(SolutionDatabase database, IndexStorage storage) {
        this.database = database;
        this.storage = storage;
    }

    @SafeVarargs
    public final void indexDiffs(String indexName, EncodingStrategy<AtomicChange>... strategies) {
        indexDiffs(indexName, Arrays.asList(strategies));
    }

    public Map<Long, Long> indexDiffs(String indexName, List<EncodingStrategy<AtomicChange>> strategies) {
        final Map<Long, Long> index = new HashMap<>();
        for (EncodingStrategy<AtomicChange> strategy : strategies) {
            final Map<Long, Long> buffer = Utils.toStream(database.iterateChanges())
                    .map(strategy::encode)
                    .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
            for (Entry<Long, Long> entry : buffer.entrySet()) {
                final Long old = index.putIfAbsent(entry.getKey(), entry.getValue());
                if (old != null && !old.equals(entry.getValue())) {
                    throw new RuntimeException("Conflict encoding!");
                }
            }
            System.out.println("Size: " + buffer.size());
        }
        storage.dropIndex(indexName);
        storage.saveIndex(indexName, index);
        return index;
    }
}
