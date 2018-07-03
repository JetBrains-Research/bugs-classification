package org.ml_methods_group.core;

import org.ml_methods_group.core.Solution.Verdict;
import org.ml_methods_group.core.changes.AtomicChange;

import java.util.Iterator;
import java.util.List;

public interface SolutionDatabase extends AutoCloseable, Iterable<SolutionDiff> {
    Solution findBySession(int sessionId, Verdict verdict);
    List<SolutionDiff> findByProblem(int problem);
    List<SolutionDiff> getDiffs();
    Iterator<AtomicChange> iterateChanges();
    SolutionDiff getDiff(int session);
    void insertSolution(Solution solution);
    void insertSolutionDiff(SolutionDiff solutionDiff);
    void clear();
    void create();
}
