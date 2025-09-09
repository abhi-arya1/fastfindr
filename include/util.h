#include <iostream>
#include <onnxruntime_cxx_api.h>


struct RuntimeOptions {
    OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_INFO;
    bool use_cuda = false;
};

RuntimeOptions parse_runtime_options(int argc, char** argv) {
    RuntimeOptions options;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--level") {
            if (i + 1 < argc) {
                int level = std::stoi(argv[++i]);
                switch (level) {
                    case 1: 
                        options.logging_level = ORT_LOGGING_LEVEL_WARNING;
                        break;
                    case 2:
                        options.logging_level = ORT_LOGGING_LEVEL_INFO;
                        break;
                    case 3:
                        options.logging_level = ORT_LOGGING_LEVEL_VERBOSE;
                        break;
                    default:
                        std::cout << "Invalid log level. Using default (INFO)." << std::endl;
                }
            }
        }
    }

    return options;
}
