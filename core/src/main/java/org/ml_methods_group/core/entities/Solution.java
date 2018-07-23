package org.ml_methods_group.core.entities;

import org.ml_methods_group.core.database.annotations.BinaryFormat;
import org.ml_methods_group.core.database.annotations.DataClass;
import org.ml_methods_group.core.database.annotations.DataField;

@DataClass(defaultStorageName = "solutions")
public class Solution {
    @BinaryFormat
    @DataField
    private final String code;
    @DataField
    private final int problemId;
    @DataField
    private final int sessionId;
    @DataField
    private final int solutionId;
    @DataField
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

    public enum Verdict {OK, FAIL, UNKNOWN}
}
