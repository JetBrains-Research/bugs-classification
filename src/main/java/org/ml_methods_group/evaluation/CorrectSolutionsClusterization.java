package org.ml_methods_group.evaluation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.clustering.clusterers.ClusterSizeLimitedHAC;
import org.ml_methods_group.clustering.clusterers.HAC;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.ASTUtils;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.representatives.CentroidPicker;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;

import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.*;

public class CorrectSolutionsClusterization {

    public static String[] problems = {
            "deserialization",
            "loggers"
    };

    public static void main(String[] args) throws Exception {
        for (String problem : problems) {
            Path pathToDataset = EvaluationInfo.PATH_TO_DATASET.resolve(problem);
            final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
            final ChangeGenerator changeGenerator = new BasicChangeGenerator(astGenerator);
            final Unifier<Solution> unifier = new BasicUnifier<>(
                    CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                    CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                    new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
            final DistanceFunction<Solution> metric = new HeuristicChangesBasedDistanceFunction(changeGenerator);

            // Collect data
            final Dataset train = loadDataset(pathToDataset.resolve("train.tmp"));
            final Dataset test = loadDataset(pathToDataset.resolve("test.tmp"));
            final List<Solution> unifiedCorrect = unifier.unify(train.getValues(x -> x.getVerdict() == OK));

            // Create clusters based on correct solutions
            int threshold = 100;
            int maxClusterSize = (int) Math.round(Math.sqrt(unifiedCorrect.size()));
            final Clusterer<Solution> clusterer = new ClusterSizeLimitedHAC<>(threshold, maxClusterSize, metric);
            final Clusters<Solution> clusters = clusterer.buildClusters(unifiedCorrect);
            storeSolutionClusters(clusters, pathToDataset.resolve("sqrt-clusters-" + threshold + ".tmp"));

            System.out.println(unifiedCorrect.size());
            clusters.getClusters().stream()
                    .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
                    .collect(Collectors.toList())
                    .forEach(x -> System.out.print(x.size() + " "));
        }
    }
}
