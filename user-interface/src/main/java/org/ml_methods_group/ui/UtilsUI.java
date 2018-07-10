package org.ml_methods_group.ui;

import difflib.Delta;
import difflib.DiffUtils;
import org.ml_methods_group.core.Index;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.selection.RepresenterSelector;
import org.ml_methods_group.core.testing.ExternalTester;
import org.ml_methods_group.core.testing.ExternalTester.PairGuess;
import org.ml_methods_group.core.testing.Pair;
import org.ml_methods_group.core.vectorization.Wrapper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class UtilsUI {
    public static List<String> lines(String text) {
        return Arrays.stream(text.split("\\R"))
                .map(String::trim)
                .collect(Collectors.toList());
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

    public static void markPairs(List<Pair<SolutionDiff>> cases, ConsoleIO console,
                                 Index<Pair<Integer>, PairGuess> index) {
        console.write("Start marking pairs");
        for (Pair<SolutionDiff> pair : cases) {
            console.write("--------------------Start-marking-new-pair---------------");
            console.write("First diff");
            console.write(pair.first);
            console.write("Second diff");
            console.write(pair.second);
            console.write("Your mark:");
            final PairGuess mark = PairGuess.valueOf(console.expect("+", "=", "-"));
            index.insert(new Pair<>(pair.first.getSessionId(), pair.second.getSessionId()), mark);
        }
    }
}
