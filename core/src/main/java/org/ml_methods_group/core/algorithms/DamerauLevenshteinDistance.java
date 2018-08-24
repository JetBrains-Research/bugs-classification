package org.ml_methods_group.core.algorithms;

import java.util.Arrays;
import java.util.function.BiPredicate;

public abstract class DamerauLevenshteinDistance {
    private final int n;
    private final int m;

    private DamerauLevenshteinDistance(int n, int m) {
        this.n = n;
        this.m = m;
    }

    protected abstract boolean test(int i, int j);

    public int solve() {
        int[] prev = new int[m + 1];
        int[] current = new int[m + 1];
        int[] next = new int[m + 1];
        for (int j = 1; j <= m; j++) {
            current[j] = j;
        }
        next[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (i > 1 && j > 1 && test(i - 1, j - 2) && test(i - 2, j - 1)) {
                    next[j] = min(
                            current[j] + 1,
                            next[j - 1] + 1,
                            current[j - 1] + (test(i - 1, j - 1) ? 0 : 1),
                            prev[j - 2] + 1
                    );
                } else {
                    next[j] = min(
                            current[j] + 1,
                            next[j - 1] + 1,
                            current[j - 1] + (test(i - 1, j - 1) ? 0 : 1)
                    );
                }
            }
            int[] tmp = prev;
            prev = current;
            current = next;
            next = tmp;
            Arrays.fill(next, 0);
            next[0] = i + 1;
        }
        return current[m];
    }

    private static int min(int a, int b, int c) {
        if (b > c) {
            b = c;
        }
        return a <= b ? a : b;
    }

    private static int min(int a, int b, int c, int d) {
        if (a > b) {
            a = b;
        }
        if (c > d) {
            c = d;
        }
        return a <= c ? a : c;
    }

    public static <V> DamerauLevenshteinDistance problemFor(V[] first, V[] second, BiPredicate<V, V> equalsFunction) {
        return new DamerauLevenshteinDistance(first.length, second.length) {
            @Override
            protected boolean test(int i, int j) {
                return equalsFunction.test(first[i], second[j]);
            }
        };
    }

    public static DamerauLevenshteinDistance problemFor(int[] first, int[] second) {
        return new DamerauLevenshteinDistance(first.length, second.length) {
            @Override
            protected boolean test(int i, int j) {
                return first[i] == second[j];
            }
        };
    }

    public static DamerauLevenshteinDistance problemFor(long[] first, long[] second) {
        return new DamerauLevenshteinDistance(first.length, second.length) {
            @Override
            protected boolean test(int i, int j) {
                return first[i] == second[j];
            }
        };
    }
}
