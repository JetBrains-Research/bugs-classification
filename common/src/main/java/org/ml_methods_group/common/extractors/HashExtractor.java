package org.ml_methods_group.common.extractors;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Hashers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class HashExtractor<T> implements FeaturesExtractor<T, String> {
    public final List<FeaturesExtractor<T, String>> hashers;
    private int tokensCount;

    public HashExtractor(List<FeaturesExtractor<T, String>> hashers) {
        this.hashers = hashers;
    }

    private HashExtractor(List<FeaturesExtractor<T, String>> hashers, int tokensCount) {
        this.hashers = hashers;
        this.tokensCount = tokensCount;
    }

    public int getTokensCount() { return tokensCount; }

    @Override
    public String process(T value) {
        return hashers.stream()
                .map(hasher -> hasher.process(value))
                .collect(Collectors.joining("", "", ""));
    }

    public static <T> HashExtractorBuilder<T> builder() {
        return new HashExtractorBuilder<>();
    }

    public static class HashExtractorBuilder<T> {
        final List<FeaturesExtractor<T, String>> hashers = new ArrayList<>();
        private int tokensCount = 0;

        private HashExtractorBuilder() {
        }

        public <F> HashExtractorBuilder<T> hashComponent(FeaturesExtractor<T, F> extractor,
                                                         FeaturesExtractor<? super F, String> hasher) {
            hashers.add(extractor.compose(hasher));
            return this;
        }

        public <F> HashExtractorBuilder<T> hashComponent(FeaturesExtractor<T, F> extractor,
                                                         HashExtractor<? super F> hasher) {
            hashers.add(extractor.compose(hasher));
            tokensCount += hasher.getTokensCount();
            return this;
        }

        public <F> HashExtractorBuilder<T> hashComponent(FeaturesExtractor<T, F> extractor) {
            var currentHasher = extractor.compose(Object::toString);
            hashers.add(currentHasher.compose(token -> token.replaceAll("[,'\"]", "")));
            return this;
        }

        public <F> HashExtractorBuilder<T> hashComponents(FeaturesExtractor<T, F[]> extractor,
                                                          FeaturesExtractor<? super F, String> hasher) {
            final FeaturesExtractor<T, String> result = extractor.compose(array -> Arrays.stream(array)
                            .map(hasher::process)
                            .collect(Collectors.joining("|", "{", "}")));
            hashers.add(result);
            return this;
        }

        public <F> HashExtractorBuilder<T> append(String text) {
            hashers.add(x -> text);
            if (text.equals(Hashers.getTokensSeparator())) {
                tokensCount++;
            }
            return this;
        }

        public HashExtractor<T> build() {
            return new HashExtractor<>(hashers, tokensCount);
        }
    }
}
