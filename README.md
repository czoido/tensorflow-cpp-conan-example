# Create with Conan

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
