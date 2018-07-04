package org.ml_methods_group.ui;

import difflib.Delta;
import difflib.DiffUtils;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.selection.RepresenterSelector;
import org.ml_methods_group.core.vectorization.Wrapper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;

public class UtilsUI {
    public static List<String> lines(String text) {
        return Arrays.asList(text.split("\\R"));
    }

    public static String diff(String before, String after) {
        final StringBuilder builder = new StringBuilder();
        for (Delta<String> delta : DiffUtils.diff(lines(before), lines(after)).getDeltas()) {
            delta.getOriginal().getLines().forEach(line -> builder.append("-|   ").append(line).append("\n"));
            delta.getRevised().getLines().forEach(line -> builder.append("+|   ").append(line).append("\n"));
        }
        return builder.toString();
    }

    public static List<String> markClusters(List<List<Wrapper>> clusters, RepresenterSelector<Wrapper> selector,
                                            SolutionDatabase database, ConsoleIO console,
                                            Predicate<List<Wrapper>> accept) {
        console.write("Start marking clusters");
        final List<String> marks = new ArrayList<>();
        for (List<Wrapper> wrappers : clusters) {
            if (!accept.test(wrappers)) {
                marks.add("");
                continue;
            }
            console.write("--------------------Start-marking-new-cluster---------------");
            final List<Wrapper> samples = selector.findRepresenter(3, wrappers);
            for (Wrapper wrapper : samples) {
                console.write("Next sample");
                console.write(database.getDiff(wrapper.sessionId));
                console.write("(Press Enter to continue)");
                console.readLine(); //wait user
            }
            console.write("Your mark:");
            String mark;
            do {
                mark = console.readToken();
                console.write("Confirm mark \"" + mark + "\"? (+/-)");
            } while (console.expect("+", "-") != 0);
            marks.add(mark);
        }
        return marks;
    }
}
