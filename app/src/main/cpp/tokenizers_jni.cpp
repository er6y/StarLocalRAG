#include <jni.h>
#include <string>
#include <android/log.h>

#define LOG_TAG "HuggingfaceTokenizer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// 声明 Rust 函数
extern "C" {
    void* create_tokenizer(const char* model_type);
    void* load_tokenizer_from_file(const char* path);
    char* tokenize(void* tokenizer_ptr, const char* text);
    void free_string(char* ptr);
    void free_tokenizer(void* ptr);
}

// JNI 方法实现

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_starlocalrag_HuggingfaceTokenizerJNI_createTokenizer(
    JNIEnv* env, jclass clazz, jstring model_type) {
    
    const char* model_type_str = env->GetStringUTFChars(model_type, NULL);
    LOGI("创建分词器，类型: %s", model_type_str);
    
    void* tokenizer = create_tokenizer(model_type_str);
    env->ReleaseStringUTFChars(model_type, model_type_str);
    
    if (tokenizer == NULL) {
        LOGE("创建分词器失败");
        return 0;
    }
    
    LOGI("成功创建分词器: %p", tokenizer);
    return (jlong)tokenizer;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_starlocalrag_HuggingfaceTokenizerJNI_loadTokenizerFromFile(
    JNIEnv* env, jclass clazz, jstring path) {
    
    const char* path_str = env->GetStringUTFChars(path, NULL);
    LOGI("从文件加载分词器: %s", path_str);
    
    void* tokenizer = load_tokenizer_from_file(path_str);
    env->ReleaseStringUTFChars(path, path_str);
    
    if (tokenizer == NULL) {
        LOGE("从文件加载分词器失败");
        return 0;
    }
    
    LOGI("成功从文件加载分词器: %p", tokenizer);
    return (jlong)tokenizer;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_starlocalrag_HuggingfaceTokenizerJNI_tokenize(
    JNIEnv* env, jclass clazz, jlong tokenizer_ptr, jstring text) {
    
    if (tokenizer_ptr == 0) {
        LOGE("分词器指针为空");
        return NULL;
    }
    
    const char* text_str = env->GetStringUTFChars(text, NULL);
    LOGI("分词文本: %s", text_str);
    
    char* result = tokenize((void*)tokenizer_ptr, text_str);
    env->ReleaseStringUTFChars(text, text_str);
    
    if (result == NULL) {
        LOGE("分词失败");
        return NULL;
    }
    
    jstring jresult = env->NewStringUTF(result);
    free_string(result);
    
    return jresult;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_starlocalrag_HuggingfaceTokenizerJNI_freeTokenizer(
    JNIEnv* env, jclass clazz, jlong tokenizer_ptr) {
    
    if (tokenizer_ptr == 0) {
        LOGE("尝试释放空分词器指针");
        return;
    }
    
    LOGI("释放分词器: %p", (void*)tokenizer_ptr);
    free_tokenizer((void*)tokenizer_ptr);
}
