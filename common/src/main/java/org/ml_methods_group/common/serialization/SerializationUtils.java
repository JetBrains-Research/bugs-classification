package org.ml_methods_group.common.serialization;

import java.io.*;
import java.nio.file.Path;

class SerializationUtils {
    static void storeObject(Object object, Path path) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path.toFile());
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(object);
        }
    }

    static <T> T loadObject(Class<T> aClass, Path path) throws IOException {
        try (FileInputStream fis = new FileInputStream(path.toFile());
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            return aClass.cast(ois.readObject());
        } catch (ClassNotFoundException e) {
            throw new IOException("Unexpected object type!", e);
        }
    }
}
