package org.ml_methods_group.vectorization.index;

import org.ml_methods_group.core.IndexDatabase;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.SolutionDiff;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class LabelsIndexer {
    public static void indexLabels(SolutionDatabase solutions,
                                   IndexDatabase storage, String indexName) {
        final Map<String, Long> counters = solutions.getDiffs().stream()
                .map(SolutionDiff::getChanges)
                .flatMap(List::stream)
                .flatMap(change -> Stream.of(change.getLabel(), change.getOldLabel()))
                .filter(Objects::nonNull)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));

        storage.saveIndex(indexName, counters);
    }

    public static void generateIds(IndexDatabase storage, String labelsIndexName, String idsIndexName,
                                   BiPredicate<String, Long> skip) {
        int idGenerator = 1;
        final Map<String, Integer> ids = new HashMap<>();
        for (Map.Entry<String, Long> entry : getLabels(storage, labelsIndexName).entrySet()) {
            if (!skip.test(entry.getKey(), entry.getValue())) {
                ids.put(entry.getKey(), idGenerator++);
            }
        }
        storage.saveIndex(idsIndexName, ids);
    }

    public static Map<String, Long> getLabels(IndexDatabase storage, String indexName)  {
        return storage.loadIndex(indexName, Function.identity(), Long::parseLong);
    }

    public static Map<String, Integer> getIds(IndexDatabase storage, String indexName)  {
        return storage.loadIndex(indexName, Function.identity(), Integer::parseInt);
    }
}
