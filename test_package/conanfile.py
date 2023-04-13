import os
from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.layout import basic_layout


class tflite_exampleTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "VirtualRunEnv"
    apply_env = False
    test_type = "explicit"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def layout(self):
        basic_layout(self)

    def test(self):
        if can_run(self):
            model_path = os.path.join(self.source_folder, "../assets", "lite-model_imagenet_mobilenet_v3_large_100_224_classification_5_metadata_1.tflite")
            labels_path = os.path.join(self.source_folder, "../assets", "lite-model_imagenet_mobilenet_v3_large_100_224_classification_5_metadata_1.txt")
            image_path = os.path.join(self.source_folder, "../assets", "frog.png")
            self.run(f"tflite-example {model_path} {labels_path} {image_path} 0", env="conanrun")
