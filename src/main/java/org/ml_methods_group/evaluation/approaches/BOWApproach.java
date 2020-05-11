package org.ml_methods_group.evaluation.approaches;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.extractors.BOWExtractor;
import org.ml_methods_group.common.extractors.BOWExtractor.BOWVector;
import org.ml_methods_group.common.extractors.ManyProblemsBasedChangesExtractor;
import org.ml_methods_group.common.extractors.SparseBOWExtractor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Hashers.CODE_CHANGE_HASHERS;
import static org.ml_methods_group.common.Solution.Verdict.FAIL;

public class BOWApproach {

    public static Approach<BOWVector> getDefaultApproach(int wordsLimit,
                                                         Dataset train,
                                                         FeaturesExtractor<Solution, Changes> generator) {
        final List<CodeChange> changes = train.getValues(x -> x.getVerdict() == FAIL)
                .stream()
                .map(generator::process)
                .map(Changes::getChanges)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        final HashMap<String, Integer> dict = BOWExtractor.mostCommon(CODE_CHANGE_HASHERS, changes, wordsLimit);
        final FeaturesExtractor<Solution, BOWVector> extractor = generator.compose(
                new BOWExtractor<>(dict, CODE_CHANGE_HASHERS).extend(Changes::getChanges));
        return new Approach<>(extractor, BOWExtractor::cosineDistance, "BOW" + wordsLimit);
    }

    public static Approach<SparseBOWExtractor.SparseBOWVector> getManyProblemsBasedApproach(
                                            Map<Dataset, FeaturesExtractor<Solution, Changes>> generatorByDataset) {
        final List<CodeChange> changes = new ArrayList<>();
        final var generatorByProblemId = new HashMap<Integer, FeaturesExtractor<Solution, Changes>>();
        for (var entry : generatorByDataset.entrySet()) {
            Dataset dataset = entry.getKey();
            FeaturesExtractor<Solution, Changes> generator = entry.getValue();
            int problemId = dataset.getValues().get(0).getProblemId();
            generatorByProblemId.put(problemId, generator);
            changes.addAll(dataset.getValues(x -> x.getVerdict() == FAIL)
                    .stream()
                    .map(generator::process)
                    .map(Changes::getChanges)
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));
        }
        final HashMap<String, Integer> dict = SparseBOWExtractor.getDictionary(CODE_CHANGE_HASHERS, changes);
        FeaturesExtractor<Solution, SparseBOWExtractor.SparseBOWVector> extractor =
                new ManyProblemsBasedChangesExtractor(generatorByProblemId).compose(
                        new SparseBOWExtractor<>(dict, CODE_CHANGE_HASHERS).extend(Changes::getChanges));
        return new Approach<>(extractor, SparseBOWExtractor::cosineDistance, "SPARSE_BOW");
    }
}
