package(default_visibility = ["//hermes:__subpackages__"])

py_library(
    name = "ml_trainer",
    srcs = glob(["*.py"]),
    imports = ["."],
    deps = [
        "//frameworks/python:lib",
        "//hermes/models/python/utils/trainer/base:base_trainer",
        "//hermes/tools/py/grpc_server",
        "//hermes/tools/py/logging",
        "//hermes/tools/py/utils",
    ],
)
