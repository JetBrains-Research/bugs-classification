package org.ml_methods_group.core.basic;

import org.ml_methods_group.core.Solution;

public class BasicSolution implements Solution {
    private final String code;
    private final int problemId;
    private final int sessionId;
    private final Verdict verdict;

    public BasicSolution(String code, int problemId, int sessionId, Verdict verdict) {
        this.code = code;
        this.problemId = problemId;
        this.sessionId = sessionId;
        this.verdict = verdict;
    }

    @Override
    public String getCode() {
        return null;
    }

    @Override
    public int getProblemId() {
        return 0;
    }

    @Override
    public int getSessionId() {
        return 0;
    }

    @Override
    public Verdict getVerdict() {
        return null;
    }
}
