package org.ml_methods_group.testing;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.SolutionMarksHolder;

import java.util.*;
import java.util.stream.Collectors;

public class ValidationSplitBuilder implements Iterable<Map.Entry<Dataset, Dataset>> {
    private final Random random;
    private final Dataset dataset;
    private final SolutionMarksHolder holder;
    private final int k;

    public ValidationSplitBuilder(Random random, Dataset dataset, SolutionMarksHolder holder, int k) {
        this.random = random;
        this.dataset = dataset;
        this.holder = holder;
        this.k = k;
    }

    public ValidationSplitBuilder(Dataset dataset, SolutionMarksHolder holder, int k) {
        this(new Random(), dataset, holder, k);
    }

    @Override
    public Iterator<Map.Entry<Dataset, Dataset>> iterator() {
        final List<Integer> sessions = holder.getSolutions().stream()
                .map(Solution::getSessionId)
                .collect(Collectors.toList());
        Collections.shuffle(sessions, random);
        final List<Map.Entry<Dataset, Dataset>> buffer = new ArrayList<>();
        final int n = holder.size() / k;
        for (int i = 0; i < k; i++) {
            final Set<Integer> ids = new HashSet<>(sessions.subList(i * n, (i + 1) * n));
            final Dataset validate = dataset.filter(x -> ids.contains(x.getSessionId()));
            final Dataset train = dataset.filter(x -> !ids.contains(x.getSessionId()));
            buffer.add(Map.entry(train, validate));
        }
        return buffer.iterator();
    }
}
