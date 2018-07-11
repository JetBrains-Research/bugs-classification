package org.ml_methods_group.database;

enum DataType {
    INTEGER("INTEGER", false, int.class, Integer.class),
    DOUBLE("FLOAT", false, double.class, Double.class),
    LONG("BIGINT", false, long.class, Long.class),
    STRING("TEXT", false, String.class),
    ENUM("TEXT", false),
    SERIAL("SERIAL", true),
    BYTE_ARRAY("BYTEA", false);

    private final Class[] supported;
    public final String sqlName;
    public final boolean service;

    DataType(String sqlName, boolean service, Class... supported) {
        this.sqlName = sqlName;
        this.service = service;
        this.supported = supported;
    }

    public static DataType getDefaultTypeFor(Class<?> template) {
        for (DataType dataType : values()) {
            for (Class supported : dataType.supported) {
                if (supported.equals(template)) {
                    return dataType;
                }
            }
        }
        if (template.isAssignableFrom(Enum.class)) {
            return ENUM;
        }
        return null;
    }
}
