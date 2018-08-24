package org.ml_methods_group.core.basic.validators;

import org.ml_methods_group.core.Validator;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.Scanner;

public abstract class AbstractManualValidator<V, M> implements Validator<V, M> {

    private final Scanner input;
    private final PrintWriter output;

    AbstractManualValidator(InputStream input, OutputStream output) {
        this.input = new Scanner(input);
        this.output = new PrintWriter(output);
    }

    @Override
    public boolean isValid(V value, M mark) {
        output.println("Mark validation requested for element:");
        output.println(valueToString(value));
        output.println("Suggested mark:" + markToString(mark));
        while (true) {
            output.println("Is mark acceptable? (+/-)");
            output.flush();
            switch (input.next()) {
                case "+":
                    return true;
                case "-":
                    return false;
                default:
                    output.println("Unexpected input!");
                    output.flush();
            }
        }
    }

    protected abstract String valueToString(V value);

    protected abstract String markToString(M mark);
}
