package org.ml_methods_group.core;

import org.ml_methods_group.core.basic.BasicSolution;
import org.ml_methods_group.core.basic.BasicSolutionDiff;
import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.changes.ChangeType;
import org.ml_methods_group.core.preparation.*;
import org.ml_methods_group.core.testing.ExternalTester;
import org.ml_methods_group.core.testing.ExternalTester.PairGuess;
import org.ml_methods_group.core.testing.Pair;
import org.ml_methods_group.core.vectorization.*;
import org.ml_methods_group.core.vectorization.VectorTemplate.Postprocessor;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import static org.ml_methods_group.core.Solution.Verdict.OK;
import static org.ml_methods_group.core.changes.ChangeType.*;
import static org.ml_methods_group.core.vectorization.BasicEncodingStrategy.ChangeAttribute.*;

public class Utils {
    public static void loadDatabase(InputStream inputStream, SolutionDatabase database,
                                    Map<String, LabelType> dictionary,
                                    int problem, Index<LabelWrapper, Integer> index) throws IOException {
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
        final LabelsIndexer labelsIndexer = new LabelsIndexer(database);
        labelsIndexer.indexLabels(x -> true, dictionary, index);
    }

    public static VectorTemplate generateTemplate(SolutionDatabase database,
                                                                Index<ChangeCodeWrapper, Integer> index,
                                                                List<EncodingStrategy> strategies,
                                                                Postprocessor postprocessor,
                                                                int lowerBound, int upperBound) {
        final DiffIndexer diffIndexer = new DiffIndexer(database);
        diffIndexer.indexDiffs(index, strategies);
        final List<Long> codes = index.getIndex()
                .entrySet()
                .stream()
                .filter(e -> e.getValue() < upperBound)
                .filter(e -> e.getValue() >= lowerBound)
                .map(Map.Entry::getKey)
                .map(ChangeCodeWrapper::getCode)
                .collect(Collectors.toList());
        return new VectorTemplate(codes, postprocessor, strategies);
    }

    public static List<EncodingStrategy> defaultStrategies(Map<String, Integer> dictionary) {
        return Arrays.asList(
                new BasicEncodingStrategy(dictionary, 2,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE),
                        Arrays.asList(DELETE, INSERT)),
                new BasicEncodingStrategy(dictionary, 3,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, LABEL_TYPE),
                        Arrays.asList(DELETE, INSERT)),
                new BasicEncodingStrategy(dictionary, 4,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, OLD_PARENT_TYPE),
                        Collections.singletonList(MOVE)),
                new BasicEncodingStrategy(dictionary, 5,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, OLD_PARENT_TYPE),
                        Collections.singletonList(MOVE)),
                new BasicEncodingStrategy(dictionary, 6,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, OLD_PARENT_TYPE,
                                OLD_PARENT_OF_PARENT_TYPE),
                        Collections.singletonList(MOVE)),
                new BasicEncodingStrategy(dictionary, 7,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, OLD_PARENT_TYPE,
                                LABEL_TYPE),
                        Collections.singletonList(MOVE)),
                new BasicEncodingStrategy(dictionary, 8,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE,
                                OLD_PARENT_TYPE, OLD_PARENT_OF_PARENT_TYPE, LABEL_TYPE),
                        Collections.singletonList(MOVE)),
                new BasicEncodingStrategy(dictionary, 9,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, LABEL_TYPE),
                        Collections.singletonList(UPDATE)),
                new BasicEncodingStrategy(dictionary, 10,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, LABEL_TYPE, OLD_LABEL_TYPE),
                        Collections.singletonList(UPDATE)),
                new BasicEncodingStrategy(dictionary, 11,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, LABEL_TYPE, OLD_LABEL_TYPE),
                        Collections.singletonList(UPDATE)),
                new BasicEncodingStrategy(dictionary, 12,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE),
                        Arrays.asList(ChangeType.values())),
                new BasicEncodingStrategy(dictionary, 13,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE),
                        Arrays.asList(ChangeType.values())),
                new BasicEncodingStrategy(dictionary, 14,
                        Arrays.asList(CHANGE_TYPE, LABEL_TYPE),
                        Arrays.asList(ChangeType.values()))
        );
    }

    public static <T> Stream<T> toStream(Iterator<T> iterator) {
        final Iterable<T> iterable = () -> iterator;
        return StreamSupport.stream(iterable.spliterator(), false);
    }
}
