package org.ml_methods_group.server;

public class HintRequest {
    private int problem;
    private String code;

    public HintRequest() {
        problem = -1;
        code = "";
    }

    public HintRequest(int problem, String code) {
        this.problem = problem;
        this.code = code;
    }

    public int getProblem() {
        return problem;
    }

    public void setProblem(int problem) {
        this.problem = problem;
    }

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }
}
