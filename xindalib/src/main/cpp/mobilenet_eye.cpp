#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)
#include <string.h>
#include "rknn_api.h"

#include "mobilenet_eye.h"
#include "direct_texture.h"

namespace mobilenet_eye {

rknn_context ctx = 0;
bool created = false;

const int input_index = 0;      // node name "Preprocessor/sub"
const int output_index = 0;    // node name "concat"

int img_width = 0;
int img_height = 0;
int img_channels = 0;

int output_elems = 0;
int output_size = 0;

rknn_tensor_attr outputs_attr[1];

void create(int inputSize, int channel, int numClasses, char *mParamPath)
{
    img_width = inputSize;
    img_height = inputSize;
    img_channels = channel;

    output_elems = numClasses;
    output_size = output_elems * sizeof(float);;

    LOGI("try rknn_init!");

    // Load model
    FILE *fp = fopen(mParamPath, "rb");
    if(fp == NULL) {
        LOGE("fopen %s fail!\n", mParamPath);
        return;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    void *model = malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        LOGE("fread %s fail!\n", mParamPath);
        free(model);
        fclose(fp);
        return;
    }

    fclose(fp);

    // RKNN_FLAG_ASYNC_MASK: enable async mode to use NPU efficiently.
    int ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_PRIOR_MEDIUM|RKNN_FLAG_ASYNC_MASK);
    free(model);

    if(ret < 0) {
        LOGE("rknn_init fail! ret=%d\n", ret);
        return;
    }

    outputs_attr[0].index = output_index;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
    if(ret < 0) {
        LOGI("rknn_query fail! ret=%d\n", ret);
        return;
    }

    created = true;
    LOGI("rknn_init success!");
}

void destroy() {
    LOGI("rknn_destroy!");
    rknn_destroy(ctx);
}

bool run_ssd(char *inData, float *y)
{
    if(!created) {
        LOGE("run_ssd: create hasn't successful!");
        return false;
    }

    rknn_input inputs[1];
    inputs[0].index = input_index;
    inputs[0].buf = inData;
    inputs[0].size = img_width * img_height * img_channels;
    inputs[0].pass_through = false;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    int ret = rknn_inputs_set(ctx, 1, inputs);
    if(ret < 0) {
        LOGE("rknn_input_set fail! ret=%d\n", ret);
        return false;
    }

    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        LOGE("rknn_run fail! ret=%d\n", ret);
        return false;
    }

    rknn_output outputs[1];
#if 0
    outputs[0].want_float = true;
    outputs[0].is_prealloc = true;
    outputs[0].index = output_index;
    outputs[0].buf = y;
    outputs[0].size = output_size;
#else  // for workround the wrong order issue of output index.
    outputs[0].want_float = true;
    outputs[0].is_prealloc = true;
    outputs[0].index = output_index;
    outputs[0].buf = y;
    outputs[0].size = output_size;
#endif
    ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
    if(ret < 0) {
        LOGE("rknn_outputs_get fail! ret=%d\n", ret);
        return false;
    }

    rknn_outputs_release(ctx, 1, outputs);
    return true;
}


bool run_ssd(int texId, float *y)
{
    if(!created) {
        LOGE("run_ssd: create hasn't successful!");
        return false;
    }

    char *inData = gDirectTexture.requireBufferByTexId(texId);

    if (inData == nullptr) {
        LOGE("run_ssd: invalid texture, id=%d!", texId);
        return false;
    }

    rknn_input inputs[1];
    inputs[0].index = input_index;
    inputs[0].buf = inData;
    inputs[0].size = img_width * img_height * img_channels;
    inputs[0].pass_through = false;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    int ret = rknn_inputs_set(ctx, 1, inputs);

    gDirectTexture.releaseBufferByTexId(texId);

    if(ret < 0) {
        LOGE("rknn_input_set fail! ret=%d\n", ret);
        return false;
    }

    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        LOGE("rknn_run fail! ret=%d\n", ret);
        return false;
    }

    rknn_output outputs[1];
#if 0
    outputs[0].want_float = true;
    outputs[0].is_prealloc = true;
    outputs[0].index = output_index;
    outputs[0].buf = y;
    outputs[0].size = output_size;
#else  // for workround the wrong order issue of output index.
    outputs[0].want_float = true;
    outputs[0].is_prealloc = true;
    outputs[0].index = output_index;
    outputs[0].buf = y;
    outputs[0].size = output_size;
#endif
    ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
    if(ret < 0) {
        LOGE("rknn_outputs_get fail! ret=%d\n", ret);
        return false;
    }

    rknn_outputs_release(ctx, 1, outputs);
    return true;
}

bool run_ssd(const uint8_t *rgba,
             const std::vector<float> &mean,
             float *result){
    size_t plane_size = img_height * img_width;
    float *input_data = new float[img_height*img_width*img_channels];
    for (size_t i = 0; i < plane_size; i++) {
        input_data[i] = static_cast<float>(rgba[i * 4 + 2]);                   // B
        input_data[plane_size + i] = static_cast<float>(rgba[i * 4 + 1]);      // G
        input_data[2 * plane_size + i] = static_cast<float>(rgba[i * 4]);      // R
        // Alpha is discarded
        if (mean.size() == 4) {
            input_data[i] = (input_data[i]- mean[0]) * mean[3];
            input_data[plane_size + i] = (input_data[plane_size + i] - mean[1]) * mean[3];
            input_data[2 * plane_size + i] = (input_data[2 * plane_size + i] - mean[2]) * mean[3];
        }
    }

    rknn_input inputs[1];
    inputs[0].index = input_index;
    inputs[0].buf = input_data;
    inputs[0].size = img_width * img_height * img_channels * 4;
    inputs[0].pass_through = false;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = RKNN_TENSOR_NCHW;
    int ret = rknn_inputs_set(ctx, 1, inputs);
    if(ret < 0) {
        LOGE("rknn_input_set fail! ret=%d\n", ret);
        return false;
    }

    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        LOGE("rknn_run fail! ret=%d\n", ret);
        return false;
    }

    rknn_output outputs[1];

    outputs[0].want_float = true;
    outputs[0].is_prealloc = true;
    outputs[0].index = output_index;
    outputs[0].buf = result;
    outputs[0].size = output_size;

    ret = rknn_outputs_get(ctx, 1, outputs, nullptr);

    if(ret < 0) {
        LOGE("rknn_outputs_get fail! ret=%d\n", ret);
        return false;
    }

    rknn_outputs_release(ctx, 1, outputs);
    delete [] input_data;

    return true;
}

int getNumFeatures(){
    return 3;
}

}  // namespace label_image



