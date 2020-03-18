package org.ml_methods_group.evaluation.approaches;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.extractors.BOWExtractor;
import org.ml_methods_group.common.extractors.BOWExtractor.BOWVector;
import org.ml_methods_group.common.extractors.HashExtractor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Hashers.*;
import static org.ml_methods_group.common.Solution.Verdict.FAIL;

public class BOWApproach {

    public static final ApproachTemplate<BOWVector> TEMPLATE = (d, g) -> getDefaultApproach(20000, d, g);

    public static Approach<BOWVector> getDefaultApproach(int wordsLimit, Dataset train,
                                                         FeaturesExtractor<Solution, Changes> generator) {
        return getApproach(wordsLimit, train, generator, Arrays.asList(getCodeChangeHasher(weak),
                getCodeChangeHasher(javaTypes), getCodeChangeHasher(full), getCodeChangeHasher(extended),
                getCodeChangeHasher(fullExtended), getCodeChangeHasher(deepExtended)));
    }

    private static Approach<BOWVector> getApproach(int wordsLimit, Dataset train,
                                                   FeaturesExtractor<Solution, Changes> generator,
                                                   List<HashExtractor<CodeChange>> extractors) {
        final List<CodeChange> changes = train.getValues(x -> x.getVerdict() == FAIL)
                .stream()
                .map(generator::process)
                .map(Changes::getChanges)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        final HashMap<String, Integer> dict = BOWExtractor.mostCommon(extractors, changes, wordsLimit);
        final FeaturesExtractor<Solution, BOWVector> extractor = generator.compose(
                new BOWExtractor<>(dict, extractors).extend(Changes::getChanges));
        return new Approach<>(extractor, BOWExtractor::cosineDistance, "BOW" + wordsLimit);
    }
}
