#include <jni.h>
#include <string>
#include <android/log.h>

#define LOG_TAG "TokenizersStub"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// 这是一个存根库，用于在不支持的架构上提供空实现
// 实际上，在运行时会检测到这个情况并回退到 Java 实现

extern "C" {
    void free_cstring(char* ptr) {
        // 空实现，因为我们不会分配内存
        LOGI("Stub free_cstring called - this architecture is not supported");
    }
    
    // 为 JNI 添加的存根函数
    void* create_tokenizer(const char* json_config) {
        LOGI("Stub create_tokenizer called - this architecture is not supported");
        return nullptr;
    }
    
    void* load_tokenizer_from_file(const char* path) {
        LOGI("Stub load_tokenizer_from_file called - this architecture is not supported");
        return nullptr;
    }
    
    char* tokenize(void* tokenizer_ptr, const char* text) {
        LOGI("Stub tokenize called - this architecture is not supported");
        return nullptr;
    }
    
    void free_tokenizer(void* tokenizer_ptr) {
        LOGI("Stub free_tokenizer called - this architecture is not supported");
        // 空实现
    }
    
    void free_string(char* text) {
        LOGI("Stub free_string called - this architecture is not supported");
        // 空实现
    }
}
