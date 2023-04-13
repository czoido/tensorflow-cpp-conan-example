from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout


class tflite_exampleRecipe(ConanFile):
    name = "tflite-example"
    version = "1.0"
    package_type = "application"

    # Optional metadata
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of tflite-example package here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*"

    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("tensorflow-lite/2.10.0")
        self.requires("opencv/4.5.5")
        self.requires("libwebp/1.3.0", override=True)
        self.requires("eigen/3.4.0", override=True)

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
