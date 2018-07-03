package org.ml_methods_group.core;

import org.ml_methods_group.core.vectorization.NearestSolutionVectorizer;
import org.ml_methods_group.core.vectorization.Wrapper;

import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;

public class HintGenerator implements Serializable {
    private final NearestSolutionVectorizer vectorizer;
    private final Classifier<Wrapper> classifier;
    private final HashMap<Integer, String> hints = new HashMap<>();

    public HintGenerator(NearestSolutionVectorizer vectorizer, Classifier<Wrapper> classifier) {
        this.vectorizer = vectorizer;
        this.classifier = classifier;
    }

    public String getTip(String code) throws IOException {
        final Wrapper wrapper = new Wrapper(vectorizer.process(code), -1);
        final int cluster = classifier.classify(wrapper);
        return hints.getOrDefault(cluster, "");
    }

    public void setHint(int cluster, String hint) {
        hints.put(cluster, hint);
    }
}
