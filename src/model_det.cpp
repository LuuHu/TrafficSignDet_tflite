#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>


void softNms(std::vector<BboxWithScore> &bboxes, float iou_thre, int score_threshold)
{
    int N = bboxes.size();
    int max_score, max_pos, cur_pos;
    float weight;
    BboxWithScore tmp_bbox, index_bbox;
    for (int i = 0; i < N; ++i)
    {
        max_score = bboxes[i].score;
        max_pos = i;
        tmp_bbox = bboxes[i];
        cur_pos = i + 1;
        while (cur_pos < N)
        {
            if (max_score < bboxes[cur_pos].score)
            {
                max_score = bboxes[cur_pos].score;
                max_pos = cur_pos;
            }
            cur_pos++;
        }
        bboxes[i] = bboxes[max_pos];
        bboxes[max_pos] = tmp_bbox;
        tmp_bbox = bboxes[i];
        cur_pos = i + 1;

        while (cur_pos < N)
        {
            index_bbox = bboxes[cur_pos];
            float area = index_bbox.w * index_bbox.h;
            float iou = cal_overlap(tmp_bbox, index_bbox);
            if (iou <= 0)
            {
                cur_pos++;
                continue;
            }
            iou /= area;
            weight = iou>iou_thre ? 1-iou : 1;
            // weight = exp(-(iou * iou) / 0.5);

            bboxes[cur_pos].score *= weight;
            // std::cout<<bboxes[cur_pos].score<<" ";
            if (bboxes[cur_pos].score <= score_threshold)
            {
                bboxes[cur_pos] = bboxes[--N];
                cur_pos = cur_pos - 1;
            }
            cur_pos++;
        }
    }

    bboxes.resize(N);
}

inline float cal_overlap(const BboxWithScore &bbox1, const BboxWithScore &bbox2)
{
    float iw = (std::min(bbox1.x+bbox1.w/2., bbox2.x+bbox2.w/2.) - std::max(bbox1.x-bbox1.w/2., bbox2.x-bbox2.w/2.));
    float ih = (std::min(bbox1.y+bbox1.h/2., bbox2.y+bbox2.h/2.) - std::max(bbox1.y-bbox1.h/2., bbox2.y-bbox2.h/2.));
    return (iw>0)&&(ih>0) ? iw*ih : 0.;
}

Model_Yolo_Det::Model_Yolo_Det(std::string model_path, int conf_thres, float iou_thres, int score_thres)
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
    layer_out = output_dims->data[output_dims->size-2] ;

    std::cout << "Detection Model Input Height   : " << model_height                       << std::endl;
    std::cout << "Detection Model Input Width    : " << model_width                        << std::endl;
    std::cout << "Detection Model Input channels : " << model_channels                     << std::endl;
    std::cout << "Detection Model tensors Size   : " << interpreter_uptr->tensors_size()   << std::endl;
    std::cout << "Detection Model Nodes Size     : " << interpreter_uptr->nodes_size()     << std::endl;
    std::cout << "Detection Model Inputs         : " << interpreter_uptr->GetInputName(0)  << std::endl;
    std::cout << "Detection Model Outputs        : " << interpreter_uptr->GetOutputName(0) << std::endl;
    std::cout << "Detection Model Outputs Size   : " << layer_out                          << std::endl;

    res_r = new uint8_t[layer_out][6];
    res_t.reserve(layer_out);

    conf_threshold = conf_thres;
    iou_threshold = iou_thres;
    score_threshold = score_thres;
}


void Model_Yolo_Det::infer(std::vector<Box_Data> &box_fn_t, cv::Mat img)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    
    float sr_y = img.rows*1./model_height;
    float sr_x = img.cols*1./model_width;
    cv::resize(img, img, cv::Size(model_height, model_width));

    memcpy((void *)interpreter_uptr->typed_input_tensor<uint8_t>(0), (void *)img.data, model_width*model_height*model_channels * sizeof(uint8_t));
    
    // for (size_t h = 0; h < model_height; h++)  // CHW
    //     for (size_t w = 0; w < model_width; w++)
    //     {
    //         interpreter_uptr->typed_input_tensor<uint8_t>(0)[h*model_width*model_channels + w*model_channels + 0] = img.at<cv::Vec3b>(h, w)[2]/2;
    //         interpreter_uptr->typed_input_tensor<uint8_t>(0)[h*model_width*model_channels + w*model_channels + 1] = img.at<cv::Vec3b>(h, w)[1]/2;
    //         interpreter_uptr->typed_input_tensor<uint8_t>(0)[h*model_width*model_channels + w*model_channels + 2] = img.at<cv::Vec3b>(h, w)[0]/2;
    //     }

    auto t1 = std::chrono::high_resolution_clock::now();
    interpreter_uptr->Invoke();
    auto t2 = std::chrono::high_resolution_clock::now();

    memcpy((void *)res_r, interpreter_uptr->typed_output_tensor<uint8_t>(0), 6 * layer_out * sizeof(uint8_t));

    for (int ttt = 0; ttt < layer_out; ttt++)
    {
        // std::cout<< int(res_r[ttt][0]) << "  " <<  int(res_r[ttt][1]) << "  " << int(res_r[ttt][2]) << "  " <<  int(res_r[ttt][3]) << "  " << int(res_r[ttt][4]) << "  " <<  int(res_r[ttt][5]) << std::endl;
        if (res_r[ttt][4] > conf_threshold)
        {
            res_t.push_back((BboxWithScore){
                .x = res_r[ttt][0]*5.664, // 640/113
                .y = res_r[ttt][1]*5.664,
                .w = res_r[ttt][2]*5.664,
                .h = res_r[ttt][3]*5.664,
                .score = res_r[ttt][4]});
        }
    }
    softNms(res_t, iou_threshold, score_threshold);

    std::cout<<res_t.size()<<std::endl;
    for (auto one_box : res_t)
        box_fn_t.push_back((Box_Data) {
            .x = one_box.x*sr_x,
            .y = one_box.y*sr_y,
            .w = one_box.w*sr_x,
            .h = one_box.h*sr_y,
            .class_id = 0});

    res_t.clear();

    auto t3 = std::chrono::high_resolution_clock::now();

    auto d_preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto d_invoke = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto d_postprocess = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    std::cout << "YOLOv5s preprocess: " << d_preprocess << " ms;  invoke: " << d_invoke << " ms;  postprocess: " << d_postprocess << " us\n";
}
