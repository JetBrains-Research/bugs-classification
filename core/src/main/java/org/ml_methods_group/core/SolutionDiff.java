package org.ml_methods_group.core;

import org.ml_methods_group.core.changes.AtomicChange;

import java.util.List;

public interface SolutionDiff {
    Solution getWrongSolution();
    Solution getCorrectSolution();
    List<AtomicChange> getChanges();
}
