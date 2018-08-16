package org.ml_methods_group;

import org.ml_methods_group.classification.CompositeClassifier;
import org.ml_methods_group.classification.KPairClassifier;
import org.ml_methods_group.classification.NearestCluster;
import org.ml_methods_group.clusterization.CompositeClusterer;
import org.ml_methods_group.clusterization.HAC;
import org.ml_methods_group.core.*;
import org.ml_methods_group.core.basic.extractors.ChangesExtractor;
import org.ml_methods_group.core.basic.markers.CacheMarker;
import org.ml_methods_group.core.basic.markers.ExtrapolationMarker;
import org.ml_methods_group.core.basic.markers.ManualClusterMarker;
import org.ml_methods_group.core.basic.markers.ManualSolutionMarker;
import org.ml_methods_group.core.basic.metrics.ChangeDistanceFunction;
import org.ml_methods_group.core.basic.metrics.FuzzyListDistanceFunction;
import org.ml_methods_group.core.basic.metrics.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.core.basic.selectors.CacheOptionSelector;
import org.ml_methods_group.core.basic.selectors.ClosestPairSelector;
import org.ml_methods_group.core.basic.selectors.OptionSelector;
import org.ml_methods_group.core.basic.validators.CacheValidator;
import org.ml_methods_group.core.basic.validators.ManualSolutionValidator;
import org.ml_methods_group.core.changes.*;
import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.testing.BasicClassificationTester;
import org.ml_methods_group.core.testing.ClassificationTester;
import org.ml_methods_group.core.testing.ClassificationTestingResult;
import org.ml_methods_group.core.testing.DatasetBuilder;
import org.ml_methods_group.core.testing.DatasetBuilder.Dataset;
import org.ml_methods_group.database.SQLDatabase;

import java.io.IOException;
import java.sql.SQLException;
import java.util.List;
import java.util.function.Function;

public class Main {

    public static void main(String[] args) throws SQLException, IOException, ClassNotFoundException {
        final int problemId = Integer.parseInt(args[0]);
        final int clusterSizeBound = Integer.parseInt(args[1]);
        final int testDatasetSize = Integer.parseInt(args[2]);
        final int seed = Integer.parseInt(args[3]);

        final SQLDatabase database = new SQLDatabase();

        final Repository<Solution> repository = database.getRepository(Solution.class);
        final DatasetBuilder builder = new DatasetBuilder(seed, problemId, repository);
        final Dataset dataset = builder.createDataset(builder.size() - testDatasetSize, testDatasetSize);
//        final Dataset dataset = builder.createDataset(100, testDatasetSize);


//        dataset.getTestIncorrect().forEach(marker::mark);
//        Collections.shuffle(dataset.getTrainIncorrect());
//        dataset.getTrainIncorrect().subList(0, 100)
//                .forEach(marker::mark);

        final ChangeGenerator generator = new BasicChangeGenerator(new BasicASTNormalizer());
        final DistanceFunction<Solution> metric =
                new HeuristicChangesBasedDistanceFunction(generator);
        final DistanceFunction<Changes> diffMetric = CommonUtils.metricFor(
                new FuzzyListDistanceFunction<>(new ChangeDistanceFunction(), ChangeDistanceFunction::getChangeClass),
                Changes::getChanges);
        final OptionSelector<Solution, Solution> selector = new CacheOptionSelector<>(
                new ClosestPairSelector<>(dataset.getTrainCorrect(), metric),
                database,
                Solution::getSolutionId,
                Solution::getSolutionId);
        final ChangesExtractor extractor =
                new ChangesExtractor(generator, selector);
        final Clusterer<Solution> clusterer = new CompositeClusterer<>(
                extractor,
                new HAC<Wrapper<Changes, Solution>>(0.5, 1, CommonUtils.metricFor(diffMetric, Wrapper::getFeatures)));
//        final Classifier<Solution, String> classifier = new CompositeClassifier<>(
//                extractor,
//                new NearestCluster<>(diffMetric));
//        final Classifier<Solution, String> classifier = new KPairClassifier<>(
//                dataset.getTrainCorrect(), 10, metric, Math::max, extractor, extractor,
//                new NearestCluster<>(
//                        CommonUtils.metricFor(
//                                new FuzzyListDistanceFunction<>(new ChangeDistanceFunction(),
//                                        ChangeDistanceFunction::getChangeClass),
//                                Changes::getChanges)));
        final Classifier<Solution, String> classifier = new CompositeClassifier<Solution, Changes, String>(
                extractor,
                new NearestCluster<>(
                        CommonUtils.metricFor(
                                CommonUtils.metricFor(
                                        new FuzzyListDistanceFunction<>(
                                                new ChangeDistanceFunction(),
                                                ChangeDistanceFunction::getChangeClass),
                                        Changes::getChanges),
                                Wrapper::getFeatures)));
        final Marker<List<Solution>, String> clusterMarker = new ExtrapolationMarker<>(
                new CacheMarker<>(Function.identity(), Function.identity(), Solution::getSolutionId, database, Marker.constMarker(null)), 2)
                .or(new ManualClusterMarker(5));

        final Scenario<Solution, String> scenario = new Scenario<>(clusterer, classifier, clusterMarker,
                list -> list.size() >= clusterSizeBound, dataset.getTrainIncorrect());
        scenario.run();
        final ClassificationTester<Solution, String> tester = new BasicClassificationTester<>(
                dataset.getTestIncorrect(),
                new CacheValidator<>(Function.identity(), Solution::getSolutionId,
                        new ManualSolutionValidator(), database));
        final ClassificationTestingResult<Solution, String> result = tester.test(scenario.getClassifier());
        System.out.println(result.getAccuracy());
        System.out.println(result.getCoverage(0.5));
        System.out.println(result.getPrecision(0.5));

    }
}
