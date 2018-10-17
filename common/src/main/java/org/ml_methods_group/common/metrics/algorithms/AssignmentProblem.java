package org.ml_methods_group.common.metrics.algorithms;

import java.util.Arrays;

public class AssignmentProblem {
    private final int[][] weights;
    private final int[] xyMatching;
    private final int[] yxMatching;
    private final boolean[] vx, vy;
    private final int[] maxRow;
    private final int[] minColumn;

    public AssignmentProblem(int[][] weights) {
        final int n = Math.max(weights.length, weights[0].length);
        this.xyMatching = new int[n];
        this.yxMatching = new int[n];
        this.vx = new boolean[n];
        this.vy = new boolean[n];
        this.maxRow = new int[n];
        this.minColumn = new int[n];
        this.weights = new int[n][n];
        for (int i = 0; i < weights.length; i++) {
            System.arraycopy(weights[i], 0, this.weights[i], 0, weights[i].length);
        }
    }

    public int solve() {
        Arrays.fill(xyMatching, -1);
        Arrays.fill(yxMatching, -1);
        Arrays.fill(minColumn, 0);
        for (int i = 0; i < maxRow.length; i++) {
            maxRow[i] = Arrays.stream(weights[i]).max().orElse(0);
        }
        for (int c = 0; c < weights.length; ) {
            Arrays.fill(vx, false);
            Arrays.fill(vy, false);
            int k = 0;
            for (int i = 0; i < weights.length; i++) {
                if (xyMatching[i] == -1 && tryImprove(i)) {
                    k++;
                }
            }
            c += k;
            if (k != 0) {
                continue;
            }
            int z = Integer.MAX_VALUE;
            for (int i = 0; i < weights.length; i++) {
                if (!vx[i]) {
                    continue;
                }
                for (int j = 0; j < weights.length; j++) {
                    if (!vy[j]) {
                        z = Math.min(z, maxRow[i] + minColumn[j] - weights[i][j]);
                    }
                }
            }
            for (int i = 0; i < weights.length; i++) {
                if (vx[i]) {
                    maxRow[i] -= z;
                }
                if (vy[i]) {
                    minColumn[i] += z;
                }
            }
        }
        int result = 0;
        for (int i = 0; i < weights.length; i++) {
            result += weights[i][xyMatching[i]];
        }
        return result;
    }

    private boolean tryImprove(int i) {
        if (vx[i]) {
            return false;
        }
        vx[i] = true;
        for (int j = 0; j < weights.length; j++) {
            if (weights[i][j] - maxRow[i] - minColumn[j] == 0) {
                vy[j] = true;
            }
        }
        for (int j = 0; j < weights.length; j++) {
            if (weights[i][j] - maxRow[i] - minColumn[j] == 0 && yxMatching[j] == -1) {
                xyMatching[i] = j;
                yxMatching[j] = i;
                return true;
            }
        }
        for (int j = 0; j < weights.length; j++) {
            if (weights[i][j] - maxRow[i] - minColumn[j] == 0 && tryImprove(yxMatching[j])) {
                xyMatching[i] = j;
                yxMatching[j] = i;
                return true;
            }
        }
        return false;
    }
}
