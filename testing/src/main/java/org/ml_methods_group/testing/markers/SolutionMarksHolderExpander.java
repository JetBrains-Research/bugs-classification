package org.ml_methods_group.testing.markers;

import org.ml_methods_group.common.OptionSelector;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.SolutionMarksHolder;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class SolutionMarksHolderExpander {
    private final OptionSelector<Solution, Solution> selector;
    private final ChangeGenerator changesGenerator;
    private final Scanner in;
    private final PrintStream out;

    public SolutionMarksHolderExpander(OptionSelector<Solution, Solution> selector,
                                       ChangeGenerator changesGenerator,
                                       Scanner in,
                                       PrintStream out) {
        this.selector = selector;
        this.changesGenerator = changesGenerator;
        this.in = in;
        this.out = out;
    }

    public List<String> expand(Solution solution, SolutionMarksHolder holder) {
        out.println("Next submission:");
        var closest = selector.selectOption(solution).orElseGet(Solution::new);
        out.println("Closest correct submission:");
        out.println(closest.getSessionId());
        out.println(closest.getCode());
        out.println("Submission:");
        out.println(solution.getSessionId());
        out.println(solution.getCode());
        out.println("Changes:");
        changesGenerator.getChanges(solution, closest).getChanges()
                .forEach(System.out::println);
        while (true) {
            System.out.print("Your marks:");
            final String[] marks = in.nextLine().trim().split("\\s+");
            System.out.println("Accept: " + String.join(" | ", marks) + "?");
            final String verdict = in.nextLine().trim();
            if (verdict.equals("+")) {
                Arrays.stream(marks).forEachOrdered(x -> holder.addMark(solution, x));
                return Arrays.asList(marks);
            }
        }
    }
}
