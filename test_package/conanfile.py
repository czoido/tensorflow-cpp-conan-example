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
            model_path = os.path.join(self.source_folder, "../assets", "lite-model_midas_v2_1_small_1_lite_1.tflite")
            video_path = os.path.join(self.source_folder, "../assets", "dancing.mov")
            self.run(f"tflite-example {model_path} {video_path}", env="conanrun")
