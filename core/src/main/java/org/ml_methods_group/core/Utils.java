package org.ml_methods_group.core;

import org.ml_methods_group.core.basic.BasicSolution;
import org.ml_methods_group.core.basic.BasicSolutionDiff;
import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.preparation.CSVParser;
import org.ml_methods_group.core.preparation.ChangesBuilder;
import org.ml_methods_group.core.preparation.DiffIndexer;
import org.ml_methods_group.core.preparation.LabelsIndexer;
import org.ml_methods_group.core.vectorization.ChangeEncodingStrategy;
import org.ml_methods_group.core.vectorization.EncodingStrategy;
import org.ml_methods_group.core.vectorization.VectorTemplate;
import org.ml_methods_group.core.vectorization.VectorTemplate.Postprocessor;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import static org.ml_methods_group.core.Solution.Verdict.OK;
import static org.ml_methods_group.core.vectorization.ChangeEncodingStrategy.ChangeAttribute.*;

public class Utils {
    public static void loadDatabase(InputStream inputStream, SolutionDatabase database,
                                    int problem) throws IOException {
        database.clear();
        database.create();
        final CSVParser parser = new CSVParser(inputStream);
        final ChangesBuilder builder = new ChangesBuilder();
        final Map<Integer, String> previous = new HashMap<>();
        while (parser.hasNextLine()) {
            parser.nextLine();
            if (parser.getProblemId() != problem) {
                continue;
            }
            database.insertSolution(new BasicSolution(
                    parser.getCode(),
                    parser.getProblemId(),
                    parser.getSessionId(),
                    parser.getVerdict()));
            final int sessionId = parser.getSessionId();
            if (!previous.containsKey(sessionId)) {
                previous.put(sessionId, parser.getCode());
                continue;
            }
            final String old = previous.put(sessionId, null);
            final String before = parser.getVerdict() == OK ? old : parser.getCode();
            final String after = parser.getVerdict() == OK ? parser.getCode() : old;
            database.insertSolutionDiff(new BasicSolutionDiff(
                    sessionId,
                    before,
                    after,
                    builder.findChanges(before, after)));
        }
    }

    public static String indexLabels(SolutionDatabase database, IndexStorage storage, int lowerBound, int upperBound) {
        final LabelsIndexer indexer = new LabelsIndexer(database, storage);
        indexer.indexLabels("labels", x -> true);
        final String idsName = "ids_limits_" + lowerBound + "_" + upperBound;
        indexer.generateIds("labels", idsName,
                (label, count) -> lowerBound <= count && count < upperBound);
        return idsName;
    }

    public static VectorTemplate<AtomicChange> generateTemplate(SolutionDatabase database, IndexStorage storage,
                                                         List<EncodingStrategy<AtomicChange>> strategies,
                                                         String indexName, Postprocessor postprocessor,
                                                         int lowerBound, int upperBound) {
        final Map<Long, Long> oldIndex = storage.loadIndex(indexName, Long::parseLong);
        final Map<Long, Long> index;
        if (oldIndex.isEmpty()) {
            final DiffIndexer indexer = new DiffIndexer(database, storage);
            index = indexer.indexDiffs(indexName, strategies);
        } else {
            index = oldIndex;
        }
        final List<Long> codes = index.entrySet().stream()
                .filter(e -> e.getValue() < upperBound)
                .filter(e -> e.getValue() >= lowerBound)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
        return new VectorTemplate<>(codes, postprocessor, strategies);
    }

    public static List<EncodingStrategy<AtomicChange>> defaultStrategies(Map<String, Long> dictionary) {
        return Arrays.asList(
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, PARENT_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, LABEL_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, LABEL_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, LABEL_TYPE, OLD_PARENT_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, OLD_PARENT_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, LABEL_TYPE, OLD_PARENT_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, OLD_PARENT_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, OLD_PARENT_TYPE,
                        OLD_PARENT_OF_PARENT_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, LABEL_TYPE, OLD_LABEL_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, OLD_LABEL_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, OLD_LABEL_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, LABEL_TYPE, OLD_PARENT_TYPE),
                new ChangeEncodingStrategy(dictionary,
                        CHANGE_TYPE, NODE_TYPE, LABEL_TYPE, OLD_PARENT_TYPE, OLD_PARENT_OF_PARENT_TYPE)
        );
    }

    public static <T> Stream<T> toStream(Iterator<T> iterator) {
        final Iterable<T> iterable = () -> iterator;
        return StreamSupport.stream(iterable.spliterator(), false);
    }
}
