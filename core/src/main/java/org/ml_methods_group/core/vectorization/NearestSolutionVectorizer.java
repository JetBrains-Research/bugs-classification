package org.ml_methods_group.core.vectorization;

import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.preparation.ChangesBuilder;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.List;

public class NearestSolutionVectorizer implements Serializable {
    private List<SolutionDiff> samples;
    private VectorTemplate<AtomicChange> template;
    private ChangesBuilder builder;

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

    private void writeObject(ObjectOutputStream stream) throws IOException {
        stream.writeObject(samples);
        stream.writeObject(template);
    }

    @SuppressWarnings("unchecked")
    private void readObject(ObjectInputStream stream) throws IOException, ClassNotFoundException {
        samples = (List<SolutionDiff>) stream.readObject();
        template = (VectorTemplate<AtomicChange>) stream.readObject();
        builder = new ChangesBuilder();
    }
}
