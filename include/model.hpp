
#ifndef _MODEL_HPP
#define _MODEL_HPP


#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>



struct Box_Data
{
    int x, y, w, h;
    int class_id;
};


struct BboxWithScore
{
    float x, y, w, h, score;
};

void softNms(std::vector<BboxWithScore> &bboxes, float iou_thre, int threshold);
float cal_overlap(const BboxWithScore &bbox1, const BboxWithScore &bbox2);


class Model_Yolo_Det
{
public:
    Model_Yolo_Det(std::string model_path, int conf_thres, float iou_thres, int score_thres);
    void infer(std::vector<Box_Data> &box_fn_t, cv::Mat img);

private:
    std::unique_ptr<tflite::FlatBufferModel> model_uptr;
    std::unique_ptr<tflite::Interpreter> interpreter_uptr;
    int model_height;
    int model_width;
    int model_channels;
    int layer_out = 0;


    uint8_t(*res_r)[6];
    std::vector<BboxWithScore> res_t;
    std::vector<int> output_sizes = {};
    float iou_threshold;
    int conf_threshold;
    int score_threshold;
};


class Model_Cls
{
public:
    Model_Cls(std::string model_path);
    int infer(cv::Mat img);
    
    const char* class_names[124] = {
        (char*)"wo", (char*)"po", (char*)"io", (char*)"w62", (char*)"p14", (char*)"i4l", (char*)"pm5", (char*)"pw3", (char*)"w43", (char*)"w26", (char*)"i17", (char*)"p10", (char*)"i2", (char*)"p19", (char*)"i14", (char*)"w47", (char*)"pl100", (char*)"w59", 
        (char*)"ps", (char*)"p22", (char*)"i9", (char*)"pl5", (char*)"ph4", (char*)"w46", (char*)"i10", (char*)"w34", (char*)"pl120", (char*)"i18", (char*)"p25", (char*)"pr30", (char*)"w39", (char*)"i6", (char*)"pb", (char*)"ph5", (char*)"p21", (char*)"w32", 
        (char*)"w40", (char*)"il60", (char*)"p12", (char*)"pl110", (char*)"i5", (char*)"w13", (char*)"w9", (char*)"pl50", (char*)"w42", (char*)"pl30", (char*)"pm20", (char*)"p27", (char*)"pa7", (char*)"pr50", (char*)"i16", (char*)"pg", (char*)"i13", (char*)"p13", 
        (char*)"w21", (char*)"pr40", (char*)"w57", (char*)"pl90", (char*)"pne", (char*)"pl40", (char*)"i2l", (char*)"w55", (char*)"pw2_5", (char*)"i1", (char*)"p1r", (char*)"pl60", (char*)"w68", (char*)"w31", (char*)"w3", (char*)"p16", (char*)"p9", (char*)"w30", 
        (char*)"pr60", (char*)"w56", (char*)"ph3", (char*)"pm50", (char*)"p5", (char*)"i2d", (char*)"pl20", (char*)"pn", (char*)"w66", (char*)"p8", (char*)"pl10", (char*)"pr20", (char*)"p1", (char*)"ph2_5", (char*)"p26", (char*)"p20", (char*)"ip", (char*)"i12", 
        (char*)"w45", (char*)"pm10", (char*)"i4", (char*)"w63", (char*)"pl25", (char*)"p18", (char*)"ph4_5", (char*)"i1r", (char*)"il90", (char*)"w22", (char*)"il100", (char*)"w41", (char*)"p11", (char*)"ph3_5", (char*)"pm15", (char*)"w37", (char*)"p6", (char*)"i3", 
        (char*)"pl80", (char*)"pm30", (char*)"p3", (char*)"pl15", (char*)"w24", (char*)"w58", (char*)"il80", (char*)"i11", (char*)"i2r", (char*)"pl70", (char*)"p23", (char*)"i15", (char*)"w8", (char*)"pm55", (char*)"i4d", (char*)"pa14"
        };


private:
    std::unique_ptr<tflite::FlatBufferModel> model_uptr;
    std::unique_ptr<tflite::Interpreter> interpreter_uptr;
    int model_height;
    int model_width;
    int model_channels;
    int output_size;
    int8_t *res_r;

};



#endif
