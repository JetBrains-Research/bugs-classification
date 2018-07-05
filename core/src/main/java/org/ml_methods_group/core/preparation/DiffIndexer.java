package org.ml_methods_group.core.preparation;

import org.ml_methods_group.core.Index;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.Utils;
import org.ml_methods_group.core.vectorization.ChangeCodeWrapper;
import org.ml_methods_group.core.vectorization.EncodingStrategy;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DiffIndexer {

    private final SolutionDatabase database;

    public DiffIndexer(SolutionDatabase database) {
        this.database = database;
    }

    public void indexDiffs(Index<ChangeCodeWrapper> index, List<EncodingStrategy> strategies) {
        index.clean();
        for (EncodingStrategy strategy : strategies) {
            final Map<Long, Long> buffer = Utils.toStream(database.iterateChanges())
                    .map(strategy::encode)
                    .filter(x -> x != 0)
                    .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
            for (Entry<Long, Long> entry : buffer.entrySet()) {
                index.insert(strategy.decode(entry.getKey()), entry.getValue().intValue());
            }
        }
    }
}
