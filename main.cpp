
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include "libcamera/libcamera.h"
#include <sys/mman.h>
#include <map>
#include "stdlib.h"
#include "unistd.h"
#include "pthread.h"
#include "model.hpp"


#define CAMERA_OUTPUT_HEIGHT 720
#define CAMERA_OUTPUT_WIDTH 960


using namespace libcamera;

static std::shared_ptr<Camera> camera;
 
std::mutex img_men_lock;
void *bgr_mem = malloc(CAMERA_OUTPUT_WIDTH * CAMERA_OUTPUT_HEIGHT * 3 * sizeof(uint8_t));

std::shared_ptr<Model_Yolo_Det> det_sptr;
std::shared_ptr<Model_Cls> cls_sptr;


int calibH(int x)
{
    x = x>0?x:0;
    x = x<CAMERA_OUTPUT_HEIGHT?x:CAMERA_OUTPUT_HEIGHT-1;
    return x;
}

int calibW(int x)
{
    x = x>0?x:0;
    x = x<CAMERA_OUTPUT_WIDTH?x:CAMERA_OUTPUT_WIDTH-1;
    return x;
}


static void requestComplete(Request *request)
{
    if (request->status() == Request::RequestCancelled)
        return;

    for (auto bufferPair : request->buffers())
    {
        FrameBuffer *buffer = bufferPair.second;
        // const FrameMetadata &metadata = buffer->metadata();
        // std::cout << " seq: " << std::setw(6) << std::setfill('0') << metadata.sequence << " bytesused: "<< metadata.planes()[0].bytesused << std::endl;
        const FrameBuffer::Plane &plane0 = buffer->planes()[0];
        img_men_lock.lock();
        bgr_mem = mmap(NULL, int(CAMERA_OUTPUT_WIDTH*CAMERA_OUTPUT_HEIGHT*3), PROT_READ | PROT_WRITE, MAP_SHARED, plane0.fd.get(), 0);
        img_men_lock.unlock();
    }
    request->reuse(Request::ReuseBuffers);
    camera->queueRequest(request);
}


void* run_frames(void*)
{
    std::vector<Box_Data> box_2s;
    box_2s.reserve(30);
    cv::Mat img_s;
    while (1)
    {
        img_men_lock.lock();
        cv::Mat mat2o(CAMERA_OUTPUT_HEIGHT, CAMERA_OUTPUT_WIDTH, CV_8UC3, bgr_mem);
        img_s = mat2o.clone();
        img_men_lock.unlock();
        
        det_sptr->infer(box_2s, img_s);
        for (auto &bx_s: box_2s)
        {
            cv::Rect rect_r(cv::Point(calibW(bx_s.x-bx_s.w/2), calibH(bx_s.y-bx_s.h/2)), cv::Point(calibW(bx_s.x+bx_s.w/2), calibH(bx_s.y+bx_s.h/2)));
            cv::Mat one_sign(img_s, rect_r);
            bx_s.class_id = cls_sptr->infer(one_sign);
        }

        for (auto &bx_s: box_2s)
        {
            cv::Rect rect_r(cv::Point(calibW(bx_s.x-bx_s.w/2), calibH(bx_s.y-bx_s.h/2)), cv::Point(calibW(bx_s.x+bx_s.w/2), calibH(bx_s.y+bx_s.h/2)));
            cv::rectangle(img_s, rect_r, cv::Scalar(0,255,0), 2.5);
            cv::putText(img_s, cls_sptr->class_names[bx_s.class_id], cv::Point(int(bx_s.x-bx_s.w/2),int(bx_s.y-bx_s.h/2)), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0,0,255), 1.5);
        }
        box_2s.clear();

        cv::resize(img_s, img_s, cv::Size(800,600));
        cv::imshow("cdc", img_s);
        cv::waitKey(30);
    }
}




int main()
{
    std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
    cm->start();

    for (auto const &camera : cm->cameras())
        std::cout << camera->properties().get(properties::Model) << "  --  " + camera->id() << std::endl;
    if (cm->cameras().empty())
    {
        std::cout << "No cameras were identified on the system." << std::endl;
        cm->stop();
        return 1;
    }
    camera = cm->get(cm->cameras()[0]->id());
    camera->acquire();

    // ! configurate cam 
    std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration({StreamRole::StillCapture});
    StreamConfiguration &streamConfig = config->at(0);
    std::cout << "Default viewfinder configuration is: " << streamConfig.toString() << std::endl;
    streamConfig.size.width = CAMERA_OUTPUT_WIDTH;
    streamConfig.size.height = CAMERA_OUTPUT_HEIGHT;
    streamConfig.pixelFormat = formats::RGB888;
    streamConfig.bufferCount = 1;
    config->validate();
    if (camera->configure(config.get()))
    {
        std::cout << "CONFIGURATION FAILED!" << std::endl;
        return EXIT_FAILURE;
    }

    // ! allocate space for img
    FrameBufferAllocator *allocator = new FrameBufferAllocator(camera);
    Stream *stream = streamConfig.stream();
    allocator->allocate(stream);
    std::cout << "Allocated " << allocator->buffers(stream).size() << " buffers for stream" << std::endl;

    // ! build request
    std::unique_ptr<Request> request = camera->createRequest();

    // ! attach buffer to request
    const std::unique_ptr<FrameBuffer> &buffer = allocator->buffers(stream)[0];
    request->addBuffer(stream, buffer.get());
    camera->requestCompleted.connect(requestComplete);
    camera->start();
    camera->queueRequest(request.get());

    det_sptr = std::make_shared<Model_Yolo_Det>("../models/yolov5n-uint8.tflite", 50, 0.2, 45);
    cls_sptr = std::make_shared<Model_Cls>("../models/cls_uint8.tflite");

    cv::namedWindow("cdc", cv::WINDOW_NORMAL);
    cv::setWindowProperty("cdc", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    pthread_t th_t;
    pthread_create(&th_t, NULL, run_frames, NULL);

    sleep(1410065408);

    std::cout<<"||Here\n";

    camera->stop();
    allocator->free(stream);
    delete allocator;
    camera->release();
    camera.reset();
    cm->stop();

    return EXIT_SUCCESS;
}
