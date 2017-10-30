#pragma once

#include <algorithm>
#include <vector>
#include <functional>
#include "string"
#include "net.h"
#include "gpu.h"

TEST(model, OpenCL) 
{
    ncnn::init_OpenCL();
    ncnn::uninit_OpenCL();
}


TEST(model, squeezenet) 
{
    // init path
    char root_path[256];
    getcwd(root_path, sizeof(root_path));
    
    char* pDirPos = strstr(root_path, "build");
    *pDirPos = '\0';

    std::string image_path = std::string(root_path) + "examples\\cat.ppm";
    std::string param_path = std::string(root_path) + "examples\\squeezenet_v1.1.param";
    std::string bin_path = std::string(root_path) + "examples\\squeezenet_v1.1.bin";
    std::vector<float> cls_scores;
    ncnn::Net squeezenet;
    int topk = 3;

    // load data
    cv::Mat bgr = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
    squeezenet.load_param(param_path.c_str());
    squeezenet.load_model(bin_path.c_str());
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

    // predict
    const float mean_vals[3] = { 104.f, 117.f, 123.f };
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    cls_scores.resize(out.c);
    for (int j = 0; j < out.c; j++)
    {
        const float* prob = out.data + out.cstep * j;
        cls_scores[j] = prob[0];
    }

    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
        std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        //fprintf(stderr, "%d = %f\n", index, score);
    }

    EXPECT_NEAR(vec[0].first, 0.310460, 1E-5);
    EXPECT_EQ(vec[0].second, 287);

    EXPECT_NEAR(vec[1].first, 0.275037, 1E-5);
    EXPECT_EQ(vec[1].second, 283);

    EXPECT_NEAR(vec[2].first, 0.208081, 1E-5);
    EXPECT_EQ(vec[2].second, 281);

}

