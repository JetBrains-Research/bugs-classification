package org.ml_methods_group;

import com.github.gumtreediff.actions.ActionGenerator;
import org.ml_methods_group.classification.NearestCluster;
import org.ml_methods_group.clusterization.HAC;
import org.ml_methods_group.core.*;
import org.ml_methods_group.core.basic.extractors.ChangeListExtractor;
import org.ml_methods_group.core.basic.metrics.ChangeDistanceFunction;
import org.ml_methods_group.core.basic.metrics.FuzzyListDistanceFunction;
import org.ml_methods_group.core.basic.metrics.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.core.basic.selectors.CacheTargetSelector;
import org.ml_methods_group.core.basic.selectors.ClosestTargetSelector;
import org.ml_methods_group.core.changes.BasicASTNormalizer;
import org.ml_methods_group.core.changes.BasicChangeGenerator;
import org.ml_methods_group.core.changes.ChangeGenerator;
import org.ml_methods_group.core.changes.CodeChange;
import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.database.SQLDatabase;

import java.io.IOException;
import java.sql.SQLException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.ml_methods_group.InitScript.PROBLEM;
import static org.ml_methods_group.core.entities.Solution.Verdict.FAIL;
import static org.ml_methods_group.core.entities.Solution.Verdict.OK;

public class Main {

    public static void main(String[] args) throws SQLException, IOException, ClassNotFoundException {

        final int clusterSizeBound = 5;
        final int testDatasetSize = 50;
        final int seed = 243;

        final SQLDatabase database = new SQLDatabase();
        final Repository<Solution> repository = database.getRepository(Solution.class);
        final List<Solution> correctSolutions = new ArrayList<>();
        final List<Solution> incorrectSolutions = new ArrayList<>();
        repository.get(
                repository.conditionSupplier().is("problemid", PROBLEM),
                repository.conditionSupplier().is("verdict", OK)
        ).forEachRemaining(correctSolutions::add);
        repository.get(
                repository.conditionSupplier().is("problemid", PROBLEM),
                repository.conditionSupplier().is("verdict", FAIL)
        ).forEachRemaining(incorrectSolutions::add);

        final ChangeGenerator generator = new BasicChangeGenerator(new BasicASTNormalizer());
        final TestDataset dataset = generateDataSet(correctSolutions, incorrectSolutions,
                incorrectSolutions.size() - testDatasetSize, testDatasetSize, new Random(seed));
        final DistanceFunction<Solution> metric =
                new HeuristicChangesBasedDistanceFunction(generator);
        final DistanceFunction<Wrapper<List<CodeChange>, Solution>> diffMetric
                = Wrapper.wrap(
                new FuzzyListDistanceFunction<>(new ChangeDistanceFunction(), ChangeDistanceFunction::getChangeClass)
        );
        final TargetSelector<Solution> selector = new CacheTargetSelector<>(
                database,
                "cache",
                new ClosestTargetSelector<>(metric),
                Solution::getSolutionId);
        final AnalysisStrategy<List<CodeChange>> strategy = new AnalysisStrategy<>(selector,
                new ChangeListExtractor(generator));
        dataset.trainCorrect.forEach(strategy::analyze);
        dataset.trainIncorrect.forEach(strategy::analyze);
        final List<Wrapper<List<CodeChange>, Solution>> trainWrappers =
                strategy.generateFeatures(dataset.trainIncorrect);
        final List<Wrapper<List<CodeChange>, Solution>> testWrappers =
                strategy.generateFeatures(dataset.testIncorrect);
        final HAC<Wrapper<List<CodeChange>, Solution>> hac = new HAC<>(0.4, 1,
                diffMetric);
        System.out.println("Run HAC");
        final List<List<Wrapper<List<CodeChange>, Solution>>> result = hac.buildClusters(trainWrappers);
        System.out.println(clusterSizeBound);
        System.out.println(result.size());
        System.out.println(result.stream().mapToInt(List::size).filter(x -> x >= clusterSizeBound).count());
        System.out.println(result.stream().mapToInt(List::size).filter(x -> x >= clusterSizeBound).sum());
        final Map<Wrapper<List<CodeChange>, Solution>, Integer> clusters = new HashMap<>();
        for (int i = 0; i < result.size(); i++) {
            if (result.get(i).size() >= clusterSizeBound) {
                printStats(result.get(i));
                for (Wrapper<List<CodeChange>, Solution> wrapper : result.get(i)) {
                    clusters.put(wrapper, i);
                }
            }
        }
        final Classifier<Wrapper<List<CodeChange>, Solution>> classifier = new NearestCluster<>(diffMetric);
        classifier.train(clusters);
        for (Wrapper<List<CodeChange>, Solution> t : testWrappers) {
            System.out.println("Test");
            System.out.println(t.getMeta().getSessionId());
            System.out.println(t.getMeta().getCode());
            final Solution target = selector.selectTarget(t.getMeta());
//            printDiffs(t.getMeta(), target, generator);
            final int clusterId = classifier.classify(t);
            final List<Wrapper<List<CodeChange>, Solution>> cluster = result.get(clusterId);
            System.out.println("Distances: ");
//            System.out.println("D: " + cluster.stream().mapToDouble(e -> diffMetric.distance(t, e)).average().orElse(0));
            result.stream()
                    .filter(c -> c.size() >= clusterSizeBound)
                    .mapToDouble(c -> c.stream()
                            .mapToDouble(e -> diffMetric.distance(t, e))
                            .average()
                            .orElse(0))
                    .sorted()
                    .boxed()
                    .limit(10)
                    .forEachOrdered(System.out::println);
            System.out.println();
            System.out.println("S: " + cluster.size());
            Collections.shuffle(cluster);
            cluster.stream()
                    .limit(5)
                    .forEach(x -> {
                        System.out.println("Train");
                        System.out.println(x.getMeta().getSessionId());
                        System.out.println(x.getMeta().getCode());
                    });
            System.out.println();
            System.out.println();
        }
        trainWrappers.stream()
                .map(Wrapper::getFeatures)
                .mapToInt(List::size)
                .average().ifPresent(System.out::println);
        testWrappers.stream()
                .map(Wrapper::getFeatures)
                .mapToInt(List::size)
                .average().ifPresent(System.out::println);
        for (Solution s : dataset.testIncorrect) {
            Solution g = repository.find(
                    repository.conditionSupplier().is("sessionid", s.getSessionId()),
                    repository.conditionSupplier().is("verdict", Solution.Verdict.OK)).get();
            foo(s, g, dataset.trainCorrect, generator,
                    new FuzzyListDistanceFunction<>(new ChangeDistanceFunction(), ChangeDistanceFunction::getChangeClass));
        }
    }

    public static void foo(Solution a, Solution b, List<Solution> correct, ChangeGenerator generator,
                           DistanceFunction<List<CodeChange>> m) {
        final List<CodeChange> ok = generator.getChanges(a, b);
        final List<List<CodeChange>> closest = correct.stream()
                .map(sol -> generator.getChanges(a, sol))
                .sorted(Comparator.comparingInt(List::size))
                .limit(10)
                .collect(Collectors.toList());
        System.out.println(a.getSessionId());
        System.out.println(ok.size());
        System.out.println(closest.get(0).size());
        System.out.println("Closest dist");
        closest.forEach(s -> System.out.println(m.distance(ok, s) + " " + s.size()));
        final List<Set<CodeChange>> changes = closest.stream()
                .map(HashSet::new)
                .collect(Collectors.toList());
        final Map<CodeChange, Double> res = changes.stream()
                .flatMap(Set::stream)
                .distinct()
                .collect(Collectors.toMap(Function.identity(), change -> (double) changes.stream()
                        .filter(s -> s.contains(change))
                        .count() / closest.size()));
        final List<CodeChange> common = res.entrySet().stream()
                .filter(codeChangeDoubleEntry -> codeChangeDoubleEntry.getValue() > 0.8)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
        System.out.println("Common: " + common.size());
        System.out.println(m.distance(ok, common));
        System.out.println();
        System.out.println();
    }

    public static void printDiffs(Solution first, Solution second, ChangeGenerator generator) {
            generator.getChanges(first, second)
                    .forEach(change -> {
                        final String data = Stream.of(
                                    change.getChangeType(),
                            change.getNodeType(),
                            change.getParentType(),
                            change.getParentOfParentType(),
                            change.getLabel(),
                            change.getOldParentType(),
                            change.getOldParentOfParentType(),
                            change.getOldLabel()
                    ).map(Object::toString).collect(Collectors.joining("\t"));
                    System.out.println("[] " + data);
                });
    }

    public static void printStats(List<Wrapper<List<CodeChange>, Solution>> cluster) {
        System.out.println("Cluster size: " + cluster.size());
        final List<Set<CodeChange>> changes = cluster.stream()
                .map(Wrapper::getFeatures)
                .map(HashSet::new)
                .collect(Collectors.toList());
        final Map<CodeChange, Double> res = changes.stream()
                .flatMap(Set::stream)
                .distinct()
                .collect(Collectors.toMap(Function.identity(), change -> (double) changes.stream()
                        .filter(s -> s.contains(change))
                        .count() / cluster.size()));
        cluster.stream()
                .limit(5)
                .forEach(
                        solution -> {
                            System.out.println(solution.getMeta().getSessionId());
                            System.out.println(solution.getMeta().getCode());
                        }
                );
        res.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .filter(entry -> entry.getValue() > 0)
                .forEach(entry -> {
                    final CodeChange change = entry.getKey();
                    final String data = Stream.of(
                            change.getChangeType(),
                            change.getNodeType(),
                            change.getParentType(),
                            change.getParentOfParentType(),
                            change.getLabel(),
                            change.getOldParentType(),
                            change.getOldParentOfParentType(),
                            change.getOldLabel()
                    ).map(Object::toString).collect(Collectors.joining("\t"));
                    System.out.println("[" + entry.getValue() + "] " + data);
                });
    }


    private static TestDataset generateDataSet(List<Solution> correct, List<Solution> incorrect,
                                               int trainSize, int testSize,
                                               Random random) {
        incorrect.sort(Comparator.comparingInt(Solution::getSolutionId));
        Collections.shuffle(incorrect, random);
        final List<Solution> testIncorrect = incorrect.subList(0, testSize);
        final List<Solution> trainIncorrect = incorrect.subList(testSize, testSize + trainSize);
        final Set<Integer> testSessions = testIncorrect
                .stream()
                .map(Solution::getSessionId)
                .collect(Collectors.toSet());
        final List<Solution> trainCorrect = correct
                .stream()
                .filter(solution -> !testSessions.contains(solution.getSessionId()))
                .collect(Collectors.toList());
        return new TestDataset(
                trainCorrect,
                trainIncorrect,
                testIncorrect);
    }

    private static class TestDataset {
        private final List<Solution> trainCorrect;
        private final List<Solution> trainIncorrect;
        private final List<Solution> testIncorrect;

        public TestDataset(List<Solution> trainCorrect, List<Solution> trainIncorrect, List<Solution> testIncorrect) {
            this.trainCorrect = trainCorrect;
            this.trainIncorrect = trainIncorrect;
            this.testIncorrect = testIncorrect;
        }
    }
}
