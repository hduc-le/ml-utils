package(default_visibility = ["//hermes:__subpackages__"])

py_library(
    name = "pyspark_pipeline",
    srcs = glob(["*.py"]),
    imports = ["."],
    deps = [
        "//frameworks/python:lib",
        "//hermes/models/python/utils",
        "//hermes/models/python/utils/transformer:pyspark_transformer",
        "//hermes/tools/py/logging",
        "//hermes/tools/py/utils",
    ],
)
