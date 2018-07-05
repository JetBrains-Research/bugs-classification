package org.ml_methods_group.core.preparation;

import org.ml_methods_group.core.Index;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.Utils;
import org.ml_methods_group.core.vectorization.LabelWrapper;

import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class LabelsIndexer {

    private final SolutionDatabase database;

    public LabelsIndexer(SolutionDatabase database) {
        this.database = database;
    }

    public void indexLabels(Predicate<? super String> accept, Map<String, LabelType> dictionary,
                            Index<LabelWrapper, Integer> storage) {
        storage.clean();
        final Map<String, Long> counters = Utils.toStream(database.iterateChanges())
                .flatMap(change -> Stream.of(change.getLabel(), change.getOldLabel()))
                .filter(Objects::nonNull)
                .filter(accept)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        int idGenerator = 1;
        for (Map.Entry<String, Long> entry : counters.entrySet()) {
            final LabelWrapper wrapper = new LabelWrapper(
                    entry.getKey(),
                    dictionary.getOrDefault(entry.getKey(), LabelType.UNKNOWN),
                    idGenerator++);
            storage.insert(wrapper, entry.getValue().intValue());
        }
    }
}
