#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>


Model_Cls::Model_Cls(std::string model_path)
{
    model_uptr = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    assert(model_uptr != nullptr);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_uptr.get(), resolver);
    assert(kTfLiteOk == builder(&interpreter_uptr));

    interpreter_uptr->AllocateTensors();
    interpreter_uptr->SetNumThreads(4);

    // tflite::PrintInterpreterState(interpreter_uptr.get());

    int In = interpreter_uptr->inputs()[0];
    model_height = interpreter_uptr->tensor(In)->dims->data[1];
    model_width = interpreter_uptr->tensor(In)->dims->data[2];
    model_channels = interpreter_uptr->tensor(In)->dims->data[3];

    TfLiteIntArray *output_dims = interpreter_uptr->tensor(interpreter_uptr->outputs()[0])->dims;
    output_size = output_dims->data[output_dims->size - 1];

    std::cout << "Classification Model Input Height   : " << model_height << std::endl;
    std::cout << "Classification Model Input Width    : " << model_width << std::endl;
    std::cout << "Classification Model Input Channels : " << model_channels << std::endl;
    std::cout << "Classification Model Tensors Size   : " << interpreter_uptr->tensors_size() << "\n";
    std::cout << "Classification Model Nodes Size     : " << interpreter_uptr->nodes_size() << "\n";
    std::cout << "Classification Model Inputs         : " << interpreter_uptr->inputs().size() << " : " << interpreter_uptr->GetInputName(0) << "\n";
    std::cout << "Classification Model Outputs        : " << interpreter_uptr->outputs().size() << " : " << interpreter_uptr->GetOutputName(0) << "\n";
    std::cout << "Classification Model Outputs Size   : " << output_size << std::endl;

    res_r = new int8_t[output_size];


}


int Model_Cls::infer(cv::Mat img)
{
    cv::resize(img, img, cv::Size(model_height, model_width));

    memcpy((void *)interpreter_uptr->typed_input_tensor<uint8_t>(0), (void *)img.data, model_width*model_height*model_channels * sizeof(uint8_t));

    // for (size_t h = 0; h < model_height; h++)
    //     for (size_t w = 0; w < model_width; w++)
    //     {
    //         interpreter_uptr->typed_input_tensor<uint8_t>(0)[h * model_width * model_channels + w * model_channels + 0] = img.at<cv::Vec3b>(h, w)[0];
    //         interpreter_uptr->typed_input_tensor<uint8_t>(0)[h * model_width * model_channels + w * model_channels + 1] = img.at<cv::Vec3b>(h, w)[1];
    //         interpreter_uptr->typed_input_tensor<uint8_t>(0)[h * model_width * model_channels + w * model_channels + 2] = img.at<cv::Vec3b>(h, w)[2];
    //     }

    auto t1 = std::chrono::high_resolution_clock::now();
    interpreter_uptr->Invoke();
    auto t2 = std::chrono::high_resolution_clock::now();

    memcpy((void *)res_r, interpreter_uptr->typed_output_tensor<uint8_t>(0), output_size * sizeof(int8_t));

    int idx_t = 0;
    int tmp_r = 0;
    for (int i = 0; i<output_size; i++)
    {
        if (res_r[i]>tmp_r)
            idx_t = i;
            tmp_r = res_r[i];
    }
    auto d_invoke = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "MobileNet invoke: " << d_invoke << " ms\n";
    return idx_t;
}
