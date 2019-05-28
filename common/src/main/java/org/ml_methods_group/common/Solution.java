package org.ml_methods_group.common;


import java.io.Serializable;

public class Solution implements Serializable {
    private final String code;
    private final int problemId;
    private final int sessionId;
    private final int solutionId;
    private final Verdict verdict;

    public Solution() {
        this(null, 0, 0, 0, null);
    }

    public Solution(String code, int problemId, int sessionId, int solutionId, Verdict verdict) {
        this.code = code;
        this.problemId = problemId;
        this.sessionId = sessionId;
        this.solutionId = solutionId;
        this.verdict = verdict;
    }

    public String getCode() {
        return code;
    }

    public int getProblemId() {
        return problemId;
    }

    public int getSessionId() {
        return sessionId;
    }

    public Verdict getVerdict() {
        return verdict;
    }

    public int getSolutionId() {
        return solutionId;
    }

    public enum Verdict {
        OK, FAIL, UNKNOWN;

        public static Verdict valueOf(int id) {
            if (id == OK.ordinal()) {
                return OK;
            } else if (id == FAIL.ordinal()) {
                return FAIL;
            } else if (id == UNKNOWN.ordinal()) {
                return UNKNOWN;
            }
            throw new IllegalArgumentException("Unexpected enum id");
        }
    }

    @Override
    public String toString() {
        return "Solution (session id: " + sessionId +
                ", solution id: " + solutionId +
                ", verdict: " + verdict + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Solution solution = (Solution) o;

        if (problemId != solution.problemId) return false;
        if (sessionId != solution.sessionId) return false;
        if (solutionId != solution.solutionId) return false;
        if (!code.equals(solution.code)) return false;
        return verdict == solution.verdict;
    }

    @Override
    public int hashCode() {
        int result = code.hashCode();
        result = 31 * result + problemId;
        result = 31 * result + sessionId;
        result = 31 * result + solutionId;
        result = 31 * result + verdict.hashCode();
        return result;
    }
}
