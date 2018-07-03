package org.ml_methods_group.core.vectorization;

import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.preparation.ChangesBuilder;

import java.io.IOException;
import java.util.List;

public class NearestSolutionVectorizer {
    private final List<SolutionDiff> samples;
    private final VectorTemplate<AtomicChange> template;
    private final ChangesBuilder builder;

    public NearestSolutionVectorizer(List<SolutionDiff> samples, VectorTemplate<AtomicChange> template,
                                     ChangesBuilder builder) {
        this.samples = samples;
        this.template = template;
        this.builder = builder;
    }

    public double[] process(String code) throws IOException {
        int best = Integer.MAX_VALUE;
        List<AtomicChange> changes = null;
        for (SolutionDiff diff : samples) {
            final List<AtomicChange> current = builder.findChanges(code, diff.getCodeAfter());
            if (best > current.size()) {
                changes = current;
                best = current.size();
            }
        }
        return template.process(changes);
    }
}
