
#include <jni.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <pthread.h>
#include <sys/syscall.h>

#include <sched.h>
#include <vector>
#include "direct_texture.h"
#include "mobilenet_eye.h"

//typedef struct _STRUCT
//{
//    const char* msg;
//} STRUCT;

extern "C" {

static char* jstringToChar(JNIEnv* env, jstring jstr) {
    char* rtn = NULL;
    jclass clsstring = env->FindClass("java/lang/String");
    jstring strencode = env->NewStringUTF("utf-8");
    jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
    jbyteArray barr = (jbyteArray) env->CallObjectMethod(jstr, mid, strencode);
    jsize alen = env->GetArrayLength(barr);
    jbyte* ba = env->GetByteArrayElements(barr, JNI_FALSE);

    if (alen > 0) {
        rtn = new char[alen + 1];
        memcpy(rtn, ba, alen);
        rtn[alen] = 0;
    }
    env->ReleaseByteArrayElements(barr, ba, 0);
    return rtn;
}

JNIEXPORT jint JNICALL Java_xindasdk_XDSDK_init
  (JNIEnv *env, jobject obj, jint inputSize, jint channel, jint numClasses, jstring modelPath)
{
    char *mModelPath = jstringToChar(env, modelPath);
	mobilenet_eye::create(inputSize, channel, numClasses, mModelPath);
	return 0;
}

JNIEXPORT void JNICALL Java_xindasdk_XDSDK_native_1deinit
		(JNIEnv *env, jobject obj) {
	mobilenet_eye::destroy();
}

JNIEXPORT jint JNICALL Java_xindasdk_XDSDK_native_1run
  (JNIEnv *env, jobject obj, jbyteArray in, jfloatArray out) {


  	jboolean inputCopy = JNI_FALSE;
  	jbyte* const inData = env->GetByteArrayElements(in, &inputCopy);

 	jboolean outputCopy = JNI_FALSE;

  	jfloat* const y = env->GetFloatArrayElements(out, &outputCopy);

	mobilenet_eye::run_ssd((char *)inData, (float *)y);

	env->ReleaseByteArrayElements(in, inData, JNI_ABORT);
	env->ReleaseFloatArrayElements(out, y, 0);

	return 0;
}

JNIEXPORT jfloatArray JNICALL Java_xindasdk_XDSDK_geteye(JNIEnv *env, jobject obj,
                                                         jbyteArray jrgba, jint width_, jint height_, jint channel_, jlong peer) {
  	uint8_t *rgba = NULL;
  	if (NULL != jrgba) {
  	    rgba = (uint8_t *)env->GetByteArrayElements(jrgba, 0);
  	} else{
        return NULL;
  	}
  	int feature_length = mobilenet_eye::getNumFeatures();
  	float feat[feature_length];

  	std::vector<float> mean = {103.94, 116.78, 123.68, 0.017};
	if (!mobilenet_eye::run_ssd(rgba, mean, feat)) {
        return NULL;
	}

    jfloatArray jnArray = env->NewFloatArray(feature_length);
    env->SetFloatArrayRegion(jnArray, 0, feature_length, feat);

	return jnArray;
}

//JNIEXPORT jint JNICALL Java_xindasdk_XDSDK_native_1run
//  (JNIEnv *env, jobject obj, jint texId, jfloatArray out) {
//
// 	jboolean outputCopy = JNI_FALSE;
//
//  	jfloat* const y = env->GetFloatArrayElements(out, &outputCopy);
//
//	mobilenet_eye::run_ssd((int)texId, (float *)y);
//
//	env->ReleaseFloatArrayElements(out, y, 0);
//	return 0;
//}

JNIEXPORT jint JNICALL Java_xindasdk_XDSDK_native_1create_1direct_1texture
  (JNIEnv *env, jclass obj, jint width, jint height, jint fmt) {
	return (jint)gDirectTexture.createDirectTexture((int)width, (int)height, (int)fmt);
}

JNIEXPORT jboolean JNICALL Java_xindasdk_XDSDK_native_1delete_1direct_1texture
  (JNIEnv *env, jclass obj, jint texId) {
	return (jboolean)gDirectTexture.deleteDirectTexture((int)texId);
}

}