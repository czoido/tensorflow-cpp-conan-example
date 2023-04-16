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


int main(int argc, char *argv[])
{

    // model from https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/1
    // Mobile convolutional neural network for monocular depth estimation from a single RGB image.

    std::cout << argc << std::endl;
    std::string model_file = (argc < 3) ? "lite-model_midas_v2_1_small_1_lite_1.tflite" : std::string(argv[1]);

    // Video by Olia Danilevich from https://www.pexels.com/
    std::string video_file = (argc < 3) ? "dancing.mov" : std::string(argv[2]);

    auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());

    if (!model)
    {
        throw std::runtime_error("Failed to load TFLite model");
    }

    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, op_resolver)(&interpreter);

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        throw std::runtime_error("Failed to allocate tensors");
    }

    tflite::PrintInterpreterState(interpreter.get());

    auto input = interpreter->inputs()[0];
    auto input_height = interpreter->tensor(input)->dims->data[1];
    auto input_width = interpreter->tensor(input)->dims->data[2];
    auto input_channels = interpreter->tensor(input)->dims->data[3];

    cv::VideoCapture video(video_file);

    if (!video.isOpened())
    {
        std::cout << "Can't open the video: " << video_file << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (true)
    {

        video >> frame;

        if (frame.empty())
        {
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
                
        // the model uses float inputs
        cv::Mat resized_image_float;
        resized_image.convertTo(resized_image_float, CV_32FC3, 1.0 / 255.0);

        memcpy(interpreter->typed_input_tensor<float>(0), resized_image_float.data, resized_image_float.total() * resized_image_float.elemSize());

        // inference
        std::chrono::steady_clock::time_point start, end;
        start = std::chrono::steady_clock::now();
        if (interpreter->Invoke() != kTfLiteOk)
        {
            std::cerr << "Inference failed" << std::endl;
            return -1;
        }
        end = std::chrono::steady_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "ms"
                  << "--->" << processing_time << std::endl;

        float *results = interpreter->typed_output_tensor<float>(0);

        cv::Mat image(256, 256, CV_32FC1, results);

        cv::Mat normalized_image;
        cv::normalize(image, normalized_image, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        cv::namedWindow("Output Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Output Image", normalized_image);
        cv::imshow("Input Image", resized_image);


        if (cv::waitKey(10) >= 0)
        {
            break;
        }
    }

    video.release();
    cv::destroyAllWindows();

    return 0;
}
