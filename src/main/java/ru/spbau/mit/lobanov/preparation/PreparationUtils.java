package ru.spbau.mit.lobanov.preparation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.function.Function;

public class PreparationUtils {
    public static <T> void incrementCounter(Map<T, Integer> counters, T value) {
        counters.compute(value, (k, cnt) -> cnt == null ? 1 : cnt + 1);
    }

}
