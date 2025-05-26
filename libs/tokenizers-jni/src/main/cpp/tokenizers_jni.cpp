#include <jni.h>
#include <string>

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
Java_com_starlocalrag_tokenizers_TokenizerJNI_createTokenizer(
    JNIEnv* env, jclass clazz, jstring model_type) {
    
    const char* model_type_str = env->GetStringUTFChars(model_type, NULL);
    void* tokenizer = create_tokenizer(model_type_str);
    env->ReleaseStringUTFChars(model_type, model_type_str);
    
    return (jlong)tokenizer;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_starlocalrag_tokenizers_TokenizerJNI_loadTokenizerFromFile(
    JNIEnv* env, jclass clazz, jstring path) {
    
    const char* path_str = env->GetStringUTFChars(path, NULL);
    void* tokenizer = load_tokenizer_from_file(path_str);
    env->ReleaseStringUTFChars(path, path_str);
    
    return (jlong)tokenizer;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_starlocalrag_tokenizers_TokenizerJNI_tokenize(
    JNIEnv* env, jclass clazz, jlong tokenizer_ptr, jstring text) {
    
    const char* text_str = env->GetStringUTFChars(text, NULL);
    char* result = tokenize((void*)tokenizer_ptr, text_str);
    env->ReleaseStringUTFChars(text, text_str);
    
    if (result == NULL) {
        return NULL;
    }
    
    jstring jresult = env->NewStringUTF(result);
    free_string(result);
    
    return jresult;
}

extern "C" JNIEXPORT void JNICALL
Java_com_starlocalrag_tokenizers_TokenizerJNI_freeTokenizer(
    JNIEnv* env, jclass clazz, jlong tokenizer_ptr) {
    
    free_tokenizer((void*)tokenizer_ptr);
}
