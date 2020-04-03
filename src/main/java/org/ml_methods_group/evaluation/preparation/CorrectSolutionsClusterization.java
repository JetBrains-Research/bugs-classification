package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.clustering.clusterers.HAC;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.evaluation.EvaluationInfo;

import java.nio.file.Path;
import java.util.List;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.*;

public class CorrectSolutionsClusterization {

    public static String[] problems = {
            "loggers",
            "reflection",
            "factorial",
            "deserialization"
    };

    public static int[] distanceLimits = {30, 40, 50, 80, 120};

    public static void main(String[] args) throws Exception {
        for (String problem : problems) {
            Path pathToDataset = EvaluationInfo.PATH_TO_DATASET.resolve(problem);
            final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
            final ChangeGenerator changeGenerator = new BasicChangeGenerator(astGenerator);
            final DistanceFunction<Solution> metric = new HeuristicChangesBasedDistanceFunction(changeGenerator);

            // Collect data
            final Dataset train = loadDataset(pathToDataset.resolve("train.tmp"));
            final Dataset test = loadDataset(pathToDataset.resolve("test.tmp"));
            final List<Solution> correct = train.getValues(x -> x.getVerdict() == OK);
            final List<Solution> incorrect = train.getValues(x -> x.getVerdict() == FAIL);

            // Create clusters based on correct solutions
            int minClustersCount = (int) Math.round(Math.sqrt(correct.size()));
            int treshold = distanceLimits[1];
            final var clusterer = new HAC<Solution>(treshold, minClustersCount, metric);
            final Clusters<Solution> clusters = clusterer.buildClusters(correct);
            storeSolutionClusters(clusters, pathToDataset.resolve("correct-solutions-clusters-" + treshold + ".tmp"));
        }
    }
}
