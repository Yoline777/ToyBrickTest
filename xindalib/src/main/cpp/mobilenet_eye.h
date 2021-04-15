#ifndef MOBILENET_EYE
#define MOBILENET_EYE

#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "rkssd4j", ##__VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "rkssd4j", ##__VA_ARGS__);


namespace mobilenet_eye {

void create(int inputSize, int channel, int numClasses, char *mParamPath);
void destroy();
bool run_ssd(char *inData, float *y);
bool run_ssd(int texId, float *y);
bool run_ssd(const uint8_t *rgba,
             const std::vector<float> &mean,
             float *result);
int getNumFeatures();

}  // namespace label_image


#endif  //SSD_IMAGE_SSD_IMAGE_H

