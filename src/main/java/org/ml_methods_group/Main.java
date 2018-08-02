package org.ml_methods_group;

import org.ml_methods_group.classification.KNearestNeighbors;
import org.ml_methods_group.classification.NearestCluster;
import org.ml_methods_group.clusterization.HAC;
import org.ml_methods_group.core.*;
import org.ml_methods_group.core.basic.metrics.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.core.basic.extractors.ChangeListExtractor;
import org.ml_methods_group.core.basic.metrics.ListDistanceFunction;
import org.ml_methods_group.core.basic.selectors.CacheTargetSelector;
import org.ml_methods_group.core.basic.selectors.ClosestTargetSelector;
import org.ml_methods_group.core.changes.*;
import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.CachedDecision;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.database.Database;
import org.ml_methods_group.database.SQLRepository;

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
        final Database database = new Database();
        final Repository<Solution> repository = new SQLRepository<>(Solution.class, database);
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
//                900, 50, new Random(566));
                900, 50, new Random(666));
        final DistanceFunction<Solution> metric = new HeuristicChangesBasedDistanceFunction(generator);
        final DistanceFunction<Wrapper<List<CodeChange>, Solution>> diffMetric
                = Wrapper.wrap(new ListDistanceFunction<CodeChange>());
        final TargetSelector<Solution> selector = new CacheTargetSelector<>(
                new SQLRepository<>(CachedDecision.class, new Database()),
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
        final HAC<Wrapper<List<CodeChange>, Solution>> hac = new HAC<>(0.4, 100,
                diffMetric);
        final List<List<Wrapper<List<CodeChange>, Solution>>> result = hac.buildClusters(trainWrappers);
        System.out.println(result.size());
        System.out.println(result.stream().mapToInt(List::size).filter(x -> x >= 5).count());
        System.out.println(result.stream().mapToInt(List::size).filter(x -> x >= 5).sum());
        final Map<Wrapper<List<CodeChange>, Solution>, Integer> clusters = new HashMap<>();
        for (int i = 0; i < result.size(); i++) {
            if (result.get(i).size() >= 5)
                for (Wrapper<List<CodeChange>, Solution> wrapper : result.get(i)) {
                    clusters.put(wrapper, i);
                }
        }
        final Classifier<Wrapper<List<CodeChange>, Solution>> classifier = new NearestCluster<>(diffMetric);
        classifier.train(clusters);
        for (Wrapper<List<CodeChange>, Solution> t : testWrappers) {
            System.out.println("Test");
            System.out.println(t.getMeta().getSessionId());
            System.out.println(t.getMeta().getCode());
            final int clusterId = classifier.classify(t);
            final List<Wrapper<List<CodeChange>, Solution>> cluster = result.get(clusterId);
            System.out.println("D: " + cluster.stream().mapToDouble(e -> diffMetric.distance(t, e)).average().orElse(0));
            System.out.println(cluster.size());
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
    }

    public static void printStats(List<Wrapper<List<CodeChange>, Solution>> cluster) {
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
                .filter(entry -> entry.getValue() > 0.2)
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
        final Set<Integer> trainSessions = incorrect.subList(0, trainSize)
                .stream()
                .map(Solution::getSessionId)
                .collect(Collectors.toSet());
        final List<Solution> trainCorrect = correct
                .stream()
                .filter(solution -> trainSessions.contains(solution.getSessionId()))
                .collect(Collectors.toList());
        return new TestDataset(
                trainCorrect,
                incorrect.subList(0, trainSize),
                incorrect.subList(trainSize, trainSize + testSize));
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
