#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/examples/label_image/get_top_n.h>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <memory>


std::vector<std::string> read_labels(std::string labels_file) {
    std::ifstream file(labels_file.c_str());
    if (!file.is_open()) {
        std::cerr << "Can't read labels file: " << labels_file << std::endl;
        exit(-1);
    }

    std::vector<std::string> labels;

    std::string label;
    while (std::getline(file, label)) {
        labels.push_back(label);
    }
    file.close();
    return labels;
}

int main(int argc, char * argv[]) {

    // model from https://tfhub.dev/
    // A convolutional neural network model that runs on RGB images and predicts human
    // joint locations of a single person. The model is designed to be run in the browser
    // using Tensorflow.js or on devices using TF Lite in real-time, targeting
    // movement/fitness activities. This variant: MoveNet.SinglePose.Thunder is a higher
    // capacity model (compared to MoveNet.SinglePose.Lightning) that performs better
    // prediction quality while still achieving real-time (>30FPS) speed. Naturally,
    // thunder will lag behind the lightning, but it will pack a punch.

    std::string model_file = (argc>1) ? "lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite" : std::string(argv[1]);

    auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());

    if (!model) {
        throw std::runtime_error("Failed to load TFLite model");
    }

    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, op_resolver)(&interpreter);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors");
    }

    //tflite::PrintInterpreterState(interpreter.get());

    auto input = interpreter->inputs()[0];
    auto input_height = interpreter->tensor(input)->dims->data[1];
    auto input_width = interpreter->tensor(input)->dims->data[2];
    auto input_channels = interpreter->tensor(input)->dims->data[3];

    cv::Mat source_image = cv::imread(image_file);
    int image_width = source_image.size().width;
    int image_height = source_image.size().height;

    int square_dim = std::min(image_width, image_height);
    int delta_height = (image_height - square_dim) / 2;
    int delta_width = (image_width - square_dim) / 2;

    cv::Mat resized_image;

    // center + crop
    cv::resize(source_image(cv::Rect(delta_width, delta_height, square_dim, square_dim)), resized_image, cv::Size(input_width, input_height));
    
    memcpy(interpreter->typed_input_tensor<unsigned char>(0), resized_image.data, resized_image.total() * resized_image.elemSize());
    
    // inference
    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Inference failed" << std::endl;
        return -1;
    }    
    end = std::chrono::steady_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // interpret output
    int output = interpreter->outputs()[0];
    TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    std::vector<std::pair<float, int>> top_results;
    float threshold = 0.3f;

    int type = interpreter->tensor(output)->type;
    auto labels = read_labels(labels_file);
    
    tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size, 1, threshold, &top_results, kTfLiteUInt8);
    
    uint8_t* results = interpreter->typed_output_tensor<uint8_t>(0);

    std::cout << "time to process: " << processing_time << "ms" << std::endl;

    for (const auto &result : top_results)
    {
        const float confidence = result.first;
        const int index = result.second;
        std::cout << "------------------" << std::endl;
        std::cout << "detected:" + labels[index] <<  " with confidence: " << confidence << std::endl;
        std::cout << "------------------" << std::endl;
        cv::putText(source_image, labels[index], cv::Point(10, source_image.rows/8), cv::FONT_HERSHEY_SIMPLEX, 0.95, CV_RGB(255, 0, 255), 2);
    }
    
    if (show_image) {
        imshow("Display window", source_image);
        cv::waitKey(0);
    }

    return 0;
}
