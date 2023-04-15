#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/string_util.h>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <memory>

const int num_keypoints = 17;
const float confidence_threshold = 0.2;

const std::vector<std::pair<int, int>> connections = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},
    {5, 6}, {5, 7}, {7, 9}, {6, 8},
    {8, 10}, {5, 11}, {6, 12}, {11, 12},
    {11, 13}, {13, 15}, {12, 14}, {14, 16}};

void draw_keypoints(cv::Mat& resized_image, float* output) {
    // Asume que la imagen ya ha sido redimensionada a square_dim x square_dim
    int square_dim = resized_image.rows;

    // Itera sobre todos los puntos clave
    for (int i = 0; i < num_keypoints; ++i) {
        float y = output[i * 3];
        float x = output[i * 3 + 1];
        float conf = output[i * 3 + 2];

        // Si la confianza del punto clave es mayor que el umbral, dibuja el punto clave
        if (conf > confidence_threshold) {
            int img_x = static_cast<int>(x * square_dim);
            int img_y = static_cast<int>(y * square_dim);
            cv::circle(resized_image, cv::Point(img_x, img_y), 3, cv::Scalar(0, 255, 0), -1);
        }
    }

    // Itera sobre todas las conexiones y dibuja las líneas del esqueleto
    for (const auto& connection : connections) {
        int index1 = connection.first;
        int index2 = connection.second;
        float y1 = output[index1 * 3];
        float x1 = output[index1 * 3 + 1];
        float conf1 = output[index1 * 3 + 2];
        float y2 = output[index2 * 3];
        float x2 = output[index2 * 3 + 1];
        float conf2 = output[index2 * 3 + 2];

        if (conf1 > confidence_threshold && conf2 > confidence_threshold) {
            int img_x1 = static_cast<int>(x1 * square_dim);
            int img_y1 = static_cast<int>(y1 * square_dim);
            int img_x2 = static_cast<int>(x2 * square_dim);
            int img_y2 = static_cast<int>(y2 * square_dim);
            cv::line(resized_image, cv::Point(img_x1, img_y1), cv::Point(img_x2, img_y2), cv::Scalar(255, 0, 0), 2);
        }
    }
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

    std::cout << argc << std::endl;
    std::string model_file = (argc<3) ? "lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite" : std::string(argv[1]);

    // Video by Olia Danilevich from https://www.pexels.com/
    std::string video_file = (argc<3) ? "dancing.mp4" : std::string(argv[2]);

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

    tflite::PrintInterpreterState(interpreter.get());

    auto input = interpreter->inputs()[0];
    auto input_height = interpreter->tensor(input)->dims->data[1];
    auto input_width = interpreter->tensor(input)->dims->data[2];
    auto input_channels = interpreter->tensor(input)->dims->data[3];

    cv::VideoCapture video(video_file);

    if (!video.isOpened()) {
        std::cout << "Can't open the video: " << video_file << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (true) {

        video >> frame;

        if (frame.empty()) {
            video.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        int image_width = frame.size().width;
        int image_height = frame.size().height;

        int square_dim = std::min(image_width, image_height);
        int delta_height = (image_height - square_dim) / 2;
        int delta_width = (image_width - square_dim) / 2;

        cv::Mat resized_image;

        // center + crop
        cv::resize(frame(cv::Rect(delta_width, delta_height, square_dim, square_dim)), resized_image, cv::Size(input_width, input_height));
        
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
        
        std::cout << "ms" << "--->" << processing_time << std::endl;
        
        float* results = interpreter->typed_output_tensor<float>(0);

        draw_keypoints(resized_image, results);

        imshow("Output", resized_image);

        if (cv::waitKey(10) >= 0) {
            break;
        }
    }

    video.release();
    cv::destroyAllWindows();

    return 0;
}
