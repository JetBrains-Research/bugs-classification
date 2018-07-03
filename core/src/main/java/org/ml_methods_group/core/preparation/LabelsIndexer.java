package org.ml_methods_group.core.preparation;

import org.ml_methods_group.core.IndexDatabase;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.changes.AtomicChange;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class LabelsIndexer {

    private final SolutionDatabase database;
    private final IndexDatabase storage;

    public LabelsIndexer(SolutionDatabase database, IndexDatabase storage) {
        this.database = database;
        this.storage = storage;
    }

    public void indexLabels(String indexName, Predicate<String> accept) {
        final Iterator<AtomicChange> changes = database.iterateChanges();
        final Map<String, Long> counters = Stream.generate(() -> changes.hasNext() ? changes.next() : null)
                .flatMap(change -> Stream.of(change.getLabel(), change.getOldLabel()))
                .filter(Objects::nonNull)
                .filter(accept)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        storage.dropIndex(indexName);
        storage.saveIndex(indexName, counters);
    }

    public void generateIds(String labelsIndexName, String idsIndexName,
                            BiPredicate<String, Long> skip) {
        long idGenerator = 1;
        final Map<String, Long> ids = new HashMap<>();
        for (Map.Entry<String, Long> entry : storage.loadIndex(labelsIndexName).entrySet()) {
            if (!skip.test(entry.getKey(), entry.getValue())) {
                ids.put(entry.getKey(), idGenerator++);
            }
        }
        storage.dropIndex(idsIndexName);
        storage.saveIndex(idsIndexName, ids);
    }
}
