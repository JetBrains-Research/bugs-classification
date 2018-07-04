package org.ml_methods_group.ui;

import org.ml_methods_group.core.Solution;
import org.ml_methods_group.core.SolutionDiff;

import java.io.InputStream;
import java.io.PrintStream;
import java.util.Objects;
import java.util.Scanner;

public class ConsoleIO {
    private final PrintStream output;
    private final Scanner input;

    public ConsoleIO(PrintStream output, InputStream input) {
        this.output = output;
        this.input = new Scanner(input);
    }

    public ConsoleIO() {
        this(System.out, System.in);
    }

    public void write(String text) {
        output.println(text);
        output.flush();
    }

    public void write(Solution solution) {
        output.println("-------------------------------Solution #" + solution.getSessionId());
        output.println("Problem: " + solution.getProblemId() + " Verdict: " + solution.getVerdict());
        output.println(solution.getCode());
        output.println("-----------------------------------------");
        output.flush();
    }

    public void write(SolutionDiff diff) {
        output.println("-------------------------------Session #" + diff.getSessionId());
        output.println("------------------------before----------------------------");
        output.println(diff.getCodeBefore());
        output.println("------------------------after-----------------------------");
        output.println(diff.getCodeAfter());
        output.println("-------------------------diff-----------------------------");
        output.println(UtilsUI.diff(diff.getCodeBefore(), diff.getCodeAfter()));
        output.println("----------------------------------------------------------");
        output.println();
    }

    public String readLine() {
        return input.nextLine();
    }

    public String readToken() {
        return input.next();
    }

    public int readInt() {
        return input.nextInt();
    }

    public int expect(String... choices) {
        while (true) {
            final String token = readToken();
            for (int i = 0; i < choices.length; i++) {
                if (Objects.equals(token, choices[i])) {
                    return i;
                }
            }
            write("Warning: unexpected token!");
        }
    }
}
