# Create with Conan

This repo has a more updated version in the conan-io/examples2 repo: https://github.com/conan-io/examples2/tree/main/examples/libraries/tensorflow-lite/pose-estimation

Also read the related blogpost: https://blog.conan.io/2023/05/11/tensorflow-lite-cpp-mobile-ml-guide.html 

Better use with Conan 2.0

```
conan create .
```

# Run locally with Xcode

```
mkdir build && cd build
conan install .. -s build_type=Release --build=missing -c tools.cmake.cmaketoolchain:generator=Xcode
conan install .. -s build_type=Debug --build=missing -c tools.cmake.cmaketoolchain:generator=Xcode
cmake .. -G Xcode -DCMAKE_TOOLCHAIN_FILE=generators/conan_toolchain.cmake
```

open tflite-example.xcodeproj project
