package org.ml_methods_group.core;

import org.ml_methods_group.core.Solution.Verdict;

import java.util.List;
import java.util.Optional;

public interface SolutionDatabase {
    Optional<String> getProblem(int problemId);
    Solution findBySession(int sessionId, Verdict verdict);
    List<SolutionDiff> findByProblem(int problem);
    List<SolutionDiff> getDiffs();
    SolutionDiff getDiff(int session);
    void insertSolution(Solution solution);
    void clear();
    void create();
}
