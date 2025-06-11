#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <math.h>
#include <string>
#include <unistd.h>
#include <android/log.h>
#include <unordered_map>
#include <mutex>
#include "llama.h"
#include "common.h"

// Write C++ code here.
//
// Do not forget to dynamically load the C++ library into your application.
//
// For instance,
//
// In MainActivity.java:
//    static {
//       System.loadLibrary("llama-android");
//    }
//
// Or, in MainActivity.kt:
//    companion object {
//      init {
//         System.loadLibrary("llama-android")
//      }
//    }

#define TAG "llama-android.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

jclass la_int_var;
jfieldID la_int_var_value;
jmethodID la_int_var_inc;

std::string cached_token_chars;

bool is_valid_utf8(const char * string) {
    if (!string) {
        return true;
    }

    const unsigned char * bytes = (const unsigned char *)string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }

    return true;
}

static void log_callback(ggml_log_level level, const char * fmt, void * data) {
    if (level == GGML_LOG_LEVEL_ERROR)     __android_log_print(ANDROID_LOG_ERROR, TAG, fmt, data);
    else if (level == GGML_LOG_LEVEL_INFO) __android_log_print(ANDROID_LOG_INFO, TAG, fmt, data);
    else if (level == GGML_LOG_LEVEL_WARN) __android_log_print(ANDROID_LOG_WARN, TAG, fmt, data);
    else __android_log_print(ANDROID_LOG_DEFAULT, TAG, fmt, data);
}



extern "C"
JNIEXPORT jlong JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_load_1model(JNIEnv *env, jobject, jstring filename) {
    LOGi("[TRACE_POINT_1] ===== LOAD_MODEL FUNCTION ENTRY =====");
    
    llama_model_params model_params = llama_model_default_params();

    auto path_to_model = env->GetStringUTFChars(filename, 0);
    LOGi("[TRACE_POINT_1] Loading model from: %s", path_to_model);
    LOGi("[MEMORY_TRACE] Model loading started - checking memory state");

    auto model = llama_model_load_from_file(path_to_model, model_params);
    env->ReleaseStringUTFChars(filename, path_to_model);

    if (!model) {
        LOGe("load_model() failed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "load_model() failed");
        return 0;
    }

    return reinterpret_cast<jlong>(model);
}

// 新增：带GPU层数参数的模型加载方法
extern "C"
JNIEXPORT jlong JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_load_1model_1with_1gpu(JNIEnv *env, jobject, jstring filename, jint gpu_layers) {
    llama_model_params model_params = llama_model_default_params();
    
    // 设置GPU层数
    model_params.n_gpu_layers = std::max(-1, gpu_layers); // -1表示全部使用GPU，0表示仅CPU

    auto path_to_model = env->GetStringUTFChars(filename, 0);
    LOGi("Loading model from %s with %d GPU layers", path_to_model, model_params.n_gpu_layers);

    auto model = llama_model_load_from_file(path_to_model, model_params);
    env->ReleaseStringUTFChars(filename, path_to_model);

    if (!model) {
        LOGe("load_model_with_gpu() failed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "load_model_with_gpu() failed");
        return 0;
    }

    return reinterpret_cast<jlong>(model);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_free_1model(JNIEnv *, jobject, jlong model) {
    llama_model_free(reinterpret_cast<llama_model *>(model));
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_new_1context(JNIEnv *env, jobject, jlong jmodel) {
    auto model = reinterpret_cast<llama_model *>(jmodel);

    if (!model) {
        LOGe("new_context(): model cannot be null");
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Model cannot be null");
        return 0;
    }

    int n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    LOGi("Using %d threads", n_threads);

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 512;  // 设置batch大小与new_batch参数一致
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;

    llama_context * context = llama_new_context_with_model(model, ctx_params);

    if (!context) {
        LOGe("llama_new_context_with_model() returned null)");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "llama_new_context_with_model() returned null)");
        return 0;
    }

    return reinterpret_cast<jlong>(context);
}

// 新增：带参数的上下文创建方法
extern "C"
JNIEXPORT jlong JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_new_1context_1with_1params(JNIEnv *env, jobject, jlong jmodel, jint context_size, jint threads, jint gpu_layers) {
    auto model = reinterpret_cast<llama_model *>(jmodel);

    if (!model) {
        LOGe("new_context_with_params(): model cannot be null");
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Model cannot be null");
        return 0;
    }

    // 使用传入的参数，但仍然做合理性检查
    int n_threads = std::max(1, std::min(threads, (int) sysconf(_SC_NPROCESSORS_ONLN)));
    int n_ctx = std::max(512, std::min(context_size, 32768)); // 限制在合理范围内
    int n_gpu_layers = std::max(-1, gpu_layers); // -1表示全部使用GPU，0表示仅CPU
    
    LOGi("Creating context with params - ctx: %d, threads: %d, gpu_layers: %d", n_ctx, n_threads, n_gpu_layers);

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx           = n_ctx;
    ctx_params.n_batch         = 512;  // 设置batch大小与new_batch参数一致
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
    // 注意：n_gpu_layers 属于 llama_model_params，不是 llama_context_params
    // GPU层数设置应该在模型加载时进行，这里不需要设置

    llama_context * context = llama_new_context_with_model(model, ctx_params);

    if (!context) {
        LOGe("llama_new_context_with_model() returned null)");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "llama_new_context_with_model() returned null)");
        return 0;
    }

    return reinterpret_cast<jlong>(context);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_free_1context(JNIEnv *, jobject, jlong context) {
    llama_free(reinterpret_cast<llama_context *>(context));
}

extern "C"
JNIEXPORT void JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_backend_1free(JNIEnv *, jobject) {
    llama_backend_free();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_log_1to_1android(JNIEnv *, jobject) {
    llama_log_set(log_callback, NULL);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_bench_1model(
        JNIEnv *env,
        jobject,
        jlong context_pointer,
        jlong model_pointer,
        jlong batch_pointer,
        jint pp,
        jint tg,
        jint pl,
        jint nr
        ) {
    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto model = reinterpret_cast<llama_model *>(model_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);

    const int n_ctx = llama_n_ctx(context);

    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp)");

        common_batch_clear(*batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(*batch, 0, i, { 0 }, false);
        }

        batch->logits[batch->n_tokens - 1] = true;
        llama_kv_self_clear(context);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, *batch) != 0) {
            LOGi("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg)");

        llama_kv_self_clear(context);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {

            common_batch_clear(*batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(*batch, 0, i, { j }, true);
            }

            LOGi("llama_decode() text generation: %d", i);
            if (llama_decode(context, *batch) != 0) {
                LOGi("llama_decode() failed during text generation");
            }
        }

        const auto t_tg_end = ggml_time_us();

        llama_kv_self_clear(context);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));

    const auto model_size     = double(llama_model_size(model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(model)) / 1e9;

    const auto backend    = "(Android)"; // TODO: What should this be?

    std::stringstream result;
    result << std::setprecision(2);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";

    return env->NewStringUTF(result.str().c_str());
}

// 全局变量来跟踪batch的token数量，用于正确释放内存
static std::unordered_map<llama_batch*, int> batch_token_counts;
static std::unordered_map<llama_batch*, int> batch_seq_max_counts;
static std::mutex batch_map_mutex;

extern "C"
JNIEXPORT jlong JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_new_1batch(JNIEnv *env, jobject, jint n_tokens, jint embd, jint n_seq_max) {
    LOGi("[TRACE_POINT_0] ===== NEW_BATCH FUNCTION ENTRY =====");
    LOGi("[TRACE_POINT_0] new_batch called with n_tokens=%d, embd=%d, n_seq_max=%d", n_tokens, embd, n_seq_max);
    LOGi("[DEBUG] ========== new_batch START ==========");
    LOGi("[DEBUG] Input params: n_tokens=%d, embd=%d, n_seq_max=%d", n_tokens, embd, n_seq_max);

    // 参数验证
    if (n_tokens <= 0 || n_seq_max <= 0) {
        LOGe("[ERROR] Invalid batch parameters: n_tokens=%d, n_seq_max=%d", n_tokens, n_seq_max);
        return 0;
    }

    // 限制参数范围以防止过度内存分配
    if (n_tokens > 8192) {
        LOGe("[ERROR] n_tokens too large: %d (max: 8192)", n_tokens);
        return 0;
    }
    
    if (n_seq_max > 64) {
        LOGe("[ERROR] n_seq_max too large: %d (max: 64)", n_seq_max);
        return 0;
    }
    
    LOGi("[DEBUG] Parameter validation passed");

    // 使用官方Android示例的手动内存分配方式
    // Source: Copy of llama.cpp:llama_batch_init but heap-allocated.
    llama_batch *batch = new llama_batch {
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };

    // 内存对齐验证函数
    auto check_memory_alignment = [](void* ptr, const char* name, size_t expected_align = 16) {
        if (!ptr) {
            LOGe("[MEMORY_CHECK] %s allocation failed!", name);
            return false;
        }
        uintptr_t addr = (uintptr_t)ptr;
        size_t alignment = addr % expected_align;
        LOGi("[MEMORY_CHECK] %s: addr=%p, alignment=%zu bytes (addr %% %zu = %zu)", 
             name, ptr, alignment, expected_align, alignment);
        if (alignment != 0) {
            LOGe("[MEMORY_CHECK] WARNING: %s not aligned to %zu bytes!", name, expected_align);
            return false;
        }
        LOGi("[MEMORY_CHECK] %s: properly aligned to %zu bytes", name, expected_align);
        return true;
    };
    
    // 内存连续性检查函数
    auto check_memory_contiguity = [](void* ptr, size_t element_size, size_t count, const char* name) {
        if (!ptr || count == 0) return true;
        
        // 检查内存是否连续分配
        char* base = (char*)ptr;
        bool is_contiguous = true;
        
        // 简单的连续性检查：验证内存块是否在合理范围内
        uintptr_t start_addr = (uintptr_t)base;
        uintptr_t end_addr = start_addr + (element_size * count);
        size_t total_size = element_size * count;
        
        LOGi("[CONTIGUITY_CHECK] %s: start=%p, end=%p, size=%zu bytes", 
             name, (void*)start_addr, (void*)end_addr, total_size);
        
        // 检查地址范围是否合理（不应该跨越太大的内存区域）
        if (total_size > 0 && total_size < SIZE_MAX) {
            // 尝试访问第一个和最后一个元素来验证内存可访问性
            volatile char first_byte = base[0];
            volatile char last_byte = base[total_size - 1];
            LOGi("[CONTIGUITY_CHECK] %s: memory access test passed (first=0x%02x, last=0x%02x)", 
                 name, (unsigned char)first_byte, (unsigned char)last_byte);
        }
        
        // Android模拟器特殊检查：验证内存页对齐
        size_t page_size = 4096; // 典型页大小
        uintptr_t page_start = start_addr & ~(page_size - 1);
        uintptr_t page_end = (end_addr + page_size - 1) & ~(page_size - 1);
        size_t pages_used = (page_end - page_start) / page_size;
        
        LOGi("[CONTIGUITY_CHECK] %s: spans %zu pages (page_start=%p, page_end=%p)", 
             name, pages_used, (void*)page_start, (void*)page_end);
        
        return is_contiguous;
    };

    if (embd) {
        batch->embd = (float *) malloc(sizeof(float) * n_tokens * embd);
        LOGi("[DEBUG] Allocated embd array: %p (size: %d floats)", batch->embd, n_tokens * embd);
        check_memory_alignment(batch->embd, "embd array");
        check_memory_contiguity(batch->embd, sizeof(float), n_tokens * embd, "embd array");
    } else {
        batch->token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
        LOGi("[DEBUG] Allocated token array: %p (size: %d tokens)", batch->token, n_tokens);
        check_memory_alignment(batch->token, "token array");
        check_memory_contiguity(batch->token, sizeof(llama_token), n_tokens, "token array");
    }

    batch->pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens);
    batch->n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens);
    // 【关键修复】：分配 n_tokens+1 个seq_id指针，因为common_batch_add会检查seq_id[n_tokens]
    batch->seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * (n_tokens + 1));
    
    LOGi("[DEBUG] Allocated arrays: pos=%p, n_seq_id=%p, seq_id=%p", batch->pos, batch->n_seq_id, batch->seq_id);
    
    // 验证主要数组的内存对齐
    check_memory_alignment(batch->pos, "pos array");
    check_memory_alignment(batch->n_seq_id, "n_seq_id array");
    check_memory_alignment(batch->seq_id, "seq_id pointer array");
    
    // 验证主要数组的内存连续性
    check_memory_contiguity(batch->pos, sizeof(llama_pos), n_tokens, "pos array");
    check_memory_contiguity(batch->n_seq_id, sizeof(int32_t), n_tokens, "n_seq_id array");
    check_memory_contiguity(batch->seq_id, sizeof(llama_seq_id *), n_tokens + 1, "seq_id pointer array");
    
    // 为前n_tokens个位置分配seq_id数组
    for (int i = 0; i < n_tokens; ++i) {
        batch->seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
        if (i < 5) {
            LOGi("[DEBUG] Allocated seq_id[%d]: %p (size: %d seq_ids)", i, batch->seq_id[i], n_seq_max);
            check_memory_alignment(batch->seq_id[i], "seq_id sub-array");
            check_memory_contiguity(batch->seq_id[i], sizeof(llama_seq_id), n_seq_max, "seq_id sub-array");
        }
    }
    // 【关键修复】：将第n_tokens个位置设为nullptr，作为边界检查标记
    batch->seq_id[n_tokens] = nullptr;
    LOGi("[DEBUG] Set seq_id[%d] = nullptr as boundary marker", n_tokens);
    
    batch->logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens);
    
    LOGi("[DEBUG] Allocated logits array: %p (size: %d bytes)", batch->logits, n_tokens);
    check_memory_alignment(batch->logits, "logits array");
    check_memory_contiguity(batch->logits, sizeof(int8_t), n_tokens, "logits array");
    
    // 验证batch结构体本身的对齐
    check_memory_alignment(batch, "batch structure");
    
    // Android模拟器特殊检查：验证内存是否在合理范围内
    uintptr_t batch_addr = (uintptr_t)batch;
    LOGi("[ANDROID_CHECK] batch structure address: %p (0x%lx)", batch, (unsigned long)batch_addr);
    if (batch->embd) {
        uintptr_t embd_addr = (uintptr_t)batch->embd;
        LOGi("[ANDROID_CHECK] embd array address: %p (0x%lx)", batch->embd, (unsigned long)embd_addr);
    }
    if (batch->token) {
        uintptr_t token_addr = (uintptr_t)batch->token;
        LOGi("[ANDROID_CHECK] token array address: %p (0x%lx)", batch->token, (unsigned long)token_addr);
    }
    
    // 记录token数量和n_seq_max用于释放
    {
        std::lock_guard<std::mutex> lock(batch_map_mutex);
        batch_token_counts[batch] = n_tokens;
        batch_seq_max_counts[batch] = n_seq_max;
        LOGi("[DEBUG] Recorded batch %p with %d tokens and %d seq_max in map", batch, n_tokens, n_seq_max);
    }
    
    LOGi("[DEBUG] ========== new_batch END: returning %p ==========", batch);

    return reinterpret_cast<jlong>(batch);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_free_1batch(JNIEnv *, jobject, jlong batch_pointer) {
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);
    
    LOGi("[DEBUG] Freeing batch: %p", batch);
    
    if (batch) {
        int n_tokens = 0;
        int n_seq_max = 0;
        
        // 获取记录的token数量和seq_max
        {
            std::lock_guard<std::mutex> lock(batch_map_mutex);
            auto it_tokens = batch_token_counts.find(batch);
            auto it_seq_max = batch_seq_max_counts.find(batch);
            
            if (it_tokens != batch_token_counts.end()) {
                n_tokens = it_tokens->second;
                batch_token_counts.erase(it_tokens);
                LOGi("[DEBUG] Found n_tokens: %d", n_tokens);
            }
            
            if (it_seq_max != batch_seq_max_counts.end()) {
                n_seq_max = it_seq_max->second;
                batch_seq_max_counts.erase(it_seq_max);
                LOGi("[DEBUG] Found n_seq_max: %d", n_seq_max);
            }
        }
        
        // 手动释放内存，按照官方Android示例的方式
        if (batch->token) {
            free(batch->token);
            LOGi("[DEBUG] Freed token array");
        }
        if (batch->embd) {
            free(batch->embd);
            LOGi("[DEBUG] Freed embd array");
        }
        if (batch->pos) {
            free(batch->pos);
            LOGi("[DEBUG] Freed pos array");
        }
        if (batch->n_seq_id) {
            free(batch->n_seq_id);
            LOGi("[DEBUG] Freed n_seq_id array");
        }
        if (batch->seq_id) {
            // 使用官方的释放方式：遍历到nullptr结束标记
            for (int i = 0; batch->seq_id[i] != nullptr; ++i) {
                free(batch->seq_id[i]);
            }
            free(batch->seq_id);
            LOGi("[DEBUG] Freed seq_id arrays (%d tokens)", n_tokens);
        }
        if (batch->logits) {
            free(batch->logits);
            LOGi("[DEBUG] Freed logits array");
        }
        
        // 释放batch结构体
        delete batch;
        LOGi("[DEBUG] Freed batch structure");
    }
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_new_1sampler(JNIEnv *, jobject) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    return reinterpret_cast<jlong>(smpl);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_free_1sampler(JNIEnv *, jobject, jlong sampler_pointer) {
    llama_sampler_free(reinterpret_cast<llama_sampler *>(sampler_pointer));
}

extern "C"
JNIEXPORT void JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_backend_1init(JNIEnv *, jobject) {
    LOGi("[DEBUG] Initializing llama backend...");
    
    // 设置日志回调
    llama_log_set(log_callback, NULL);
    
    // 初始化后端
    llama_backend_init();
    
    LOGi("[DEBUG] Backend initialization completed");
    LOGi("[DEBUG] System info: %s", llama_print_system_info());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_system_1info(JNIEnv *env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_completion_1init(
        JNIEnv * env,
        jclass,
        jlong context_pointer,
        jlong batch_pointer,
        jstring jtext,
        jint n_len,
        jboolean format_chat
    ) {

    // 最早的跟踪点 - 确保函数被调用
    LOGi("[TRACE_POINT_2] ===== COMPLETION_INIT FUNCTION ENTRY =====");
    LOGi("[TRACE_POINT_2] Function parameters: context_ptr=%p, batch_ptr=%p, n_len=%d", 
         (void*)context_pointer, (void*)batch_pointer, n_len);
    
    cached_token_chars.clear();

    LOGi("[MEMORY_TRACE] Inference started - checking batch and context state");
    LOGi("[DEBUG] ========== completion_init START ==========");
    LOGi("[DEBUG] Input params: context_pointer=%p, batch_pointer=%p, n_len=%d, format_chat=%s",
         (void*)context_pointer, (void*)batch_pointer, n_len, format_chat ? "true" : "false");

    const auto text = env->GetStringUTFChars(jtext, 0);
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);
    
    LOGi("[DEBUG] Converted pointers: context=%p, batch=%p, text=%p", context, batch, text);

    if (!context) {
        LOGe("[ERROR] completion_init: context is null");
        env->ReleaseStringUTFChars(jtext, text);
        return -1;
    }

    if (!batch) {
        LOGe("[ERROR] completion_init: batch is null");
        env->ReleaseStringUTFChars(jtext, text);
        return -1;
    }
    
    LOGi("[DEBUG] Basic pointer validation passed");

    LOGi("[调试] 提示词内容: %s", text);

    bool parse_special = (format_chat == JNI_TRUE);
    const auto tokens_list = common_tokenize(context, text, true, parse_special);

    auto n_ctx = llama_n_ctx(context);
    auto n_batch = llama_n_batch(context);
    auto n_kv_req = tokens_list.size() + n_len;

    LOGi("[DEBUG] completion_init params: text_len=%zu, n_len=%d, format_chat=%s", strlen(text), n_len, format_chat ? "true" : "false");
    LOGi("[DEBUG] tokenized: tokens_count=%zu", tokens_list.size());
    LOGi("[DEBUG] context info: n_ctx=%d", n_ctx);
    LOGi("[DEBUG] kv_req calculation: tokens_size=%zu + n_len=%d = n_kv_req=%zu", tokens_list.size(), n_len, n_kv_req);
    LOGi("n_len = %d, n_ctx = %d, n_batch = %d, n_kv_req = %zu, tokens_count = %zu", 
         n_len, n_ctx, n_batch, n_kv_req, tokens_list.size());

    if (n_kv_req > n_ctx) {
        LOGe("error: n_kv_req > n_ctx, the required KV cache size is not big enough");
    }
    
    // 检查 token 数量是否超过 batch 大小
    if (tokens_list.size() > (size_t)n_batch) {
        LOGe("token数量(%zu)超过批处理大小(%d)，这可能导致ggml_compute_forward_transpose错误", 
             tokens_list.size(), n_batch);
        env->ReleaseStringUTFChars(jtext, text);
        return -1;
    }
    
    // 【关键修复】：检查batch的实际容量是否足够


    for (auto id : tokens_list) {
        LOGi("token: `%s`-> %d ", common_token_to_piece(context, id).c_str(), id);
    }

    // 检查 batch 结构体的完整性
    if (!batch->token || !batch->pos || !batch->n_seq_id || !batch->seq_id || !batch->logits) {
        LOGe("[ERROR] batch structure is incomplete: token=%p, pos=%p, n_seq_id=%p, seq_id=%p, logits=%p",
             batch->token, batch->pos, batch->n_seq_id, batch->seq_id, batch->logits);
        env->ReleaseStringUTFChars(jtext, text);
        return -1;
    }

    // 检查 batch 的 seq_id 数组
    for (int i = 0; i < tokens_list.size(); i++) {
        if (!batch->seq_id[i]) {
            LOGe("[ERROR] batch->seq_id[%d] is null", i);
            env->ReleaseStringUTFChars(jtext, text);
            return -1;
        }
    }

    common_batch_clear(*batch);
    LOGi("[DEBUG] batch cleared, preparing to add %zu tokens", tokens_list.size());
    LOGi("[DEBUG] batch state after clear: n_tokens=%d", batch->n_tokens);
    
    // 验证batch在clear后的状态
    if (tokens_list.size() > 0) {
        // 检查第一个位置的seq_id指针
        if (!batch->seq_id[0]) {
            LOGe("[ERROR] batch->seq_id[0] is null after clear!");
            env->ReleaseStringUTFChars(jtext, text);
            return -1;
        }
        LOGi("[DEBUG] batch->seq_id[0] = %p (verified non-null)", batch->seq_id[0]);
    }

    try {
        // evaluate the initial prompt
        for (auto i = 0; i < tokens_list.size(); i++) {
            LOGi("[DEBUG] Adding token %d: value=%d, pos=%d", i, tokens_list[i], i);
            
            // 在调用common_batch_add前验证seq_id指针
            if (!batch->seq_id[batch->n_tokens]) {
                LOGe("[ERROR] batch->seq_id[%d] is null before common_batch_add!", batch->n_tokens);
                env->ReleaseStringUTFChars(jtext, text);
                return -1;
            }
            
            LOGi("[DEBUG] About to call common_batch_add with batch->n_tokens=%d, seq_id[%d]=%p", 
                 batch->n_tokens, batch->n_tokens, batch->seq_id[batch->n_tokens]);
            
            common_batch_add(*batch, tokens_list[i], i, { 0 }, false);
            
            LOGi("[DEBUG] Successfully added token %d, batch->n_tokens now = %d", i, batch->n_tokens);
        }
    } catch (const std::exception& e) {
        LOGe("[ERROR] Exception during common_batch_add: %s", e.what());
        env->ReleaseStringUTFChars(jtext, text);
        return -1;
    } catch (...) {
        LOGe("[ERROR] Unknown exception during common_batch_add");
        env->ReleaseStringUTFChars(jtext, text);
        return -1;
    }
    
    LOGi("[DEBUG] batch prepared: n_tokens=%d", batch->n_tokens);
    
    // 添加详细的batch状态检查
    for (int i = 0; i < batch->n_tokens; i++) {
        LOGi("[DEBUG] batch[%d]: token=%d, pos=%d, seq_id[0]=%d, logits=%s", 
             i, batch->token[i], batch->pos[i], batch->seq_id[i][0], 
             batch->logits[i] ? "true" : "false");
    }
    
    // 检查batch指针和内存
    LOGi("[DEBUG] batch pointer checks: batch=%p, token=%p, pos=%p, seq_id=%p, logits=%p",
         batch, batch->token, batch->pos, batch->seq_id, batch->logits);
    
    // 检查context状态
    LOGi("[DEBUG] context checks: ctx=%p, model=%p", context, llama_get_model(context));

    // llama_decode will output logits only for the last token of the prompt
    batch->logits[batch->n_tokens - 1] = true;
    
    // 添加更详细的 batch 内容检查
    LOGi("[DEBUG] batch content validation:");
    for (int i = 0; i < batch->n_tokens && i < 5; ++i) {
        LOGi("[DEBUG] batch[%d]: token=%d, pos=%d, seq_id[0]=%d, logits=%s", 
             i, batch->token[i], batch->pos[i], 
             batch->seq_id[i] ? batch->seq_id[i][0] : -1,
             batch->logits[i] ? "true" : "false");
    }
    
    // 检查 seq_id 指针的有效性
    bool seq_id_valid = true;
    for (int i = 0; i < batch->n_tokens; ++i) {
        if (!batch->seq_id[i]) {
            LOGe("[ERROR] batch->seq_id[%d] is null!", i);
            seq_id_valid = false;
            break;
        }
    }
    
    if (!seq_id_valid) {
        LOGe("[ERROR] Invalid seq_id pointers detected, aborting decode");
        env->ReleaseStringUTFChars(jtext, text);
        return -1;
    }
    
    // 最终检查
    LOGi("[DEBUG] ========== FINAL CHECKS BEFORE llama_decode ==========");
    LOGi("[DEBUG] - batch->n_tokens=%d", batch->n_tokens);
    LOGi("[DEBUG] - last token logits set to: %s", batch->logits[batch->n_tokens - 1] ? "true" : "false");
    LOGi("[DEBUG] - context n_ctx=%d", llama_n_ctx(context));
    const auto model = llama_get_model(context);
    const auto vocab = llama_model_get_vocab(model);
    LOGi("[DEBUG] - model vocab size=%d", llama_vocab_n_tokens(vocab));
    
    // 详细的batch内存验证
    LOGi("[DEBUG] Detailed batch memory validation:");
    LOGi("[DEBUG] - batch pointer: %p", batch);
    LOGi("[DEBUG] - batch->token: %p", batch->token);
    LOGi("[DEBUG] - batch->pos: %p", batch->pos);
    LOGi("[DEBUG] - batch->seq_id: %p", batch->seq_id);
    LOGi("[DEBUG] - batch->logits: %p", batch->logits);
    
    // 验证前几个token的数据完整性
    for (int i = 0; i < batch->n_tokens && i < 3; ++i) {
        LOGi("[DEBUG] - token[%d]: value=%d, pos=%d, seq_id[0]=%d, logits=%d", 
             i, batch->token[i], batch->pos[i], 
             batch->seq_id[i] ? batch->seq_id[i][0] : -999, 
             batch->logits[i]);
        
        // 验证seq_id指针
        if (!batch->seq_id[i]) {
            LOGe("[ERROR] CRITICAL: batch->seq_id[%d] is NULL before llama_decode!", i);
            env->ReleaseStringUTFChars(jtext, text);
            return -1;
        }
    }
    
    LOGi("[DEBUG] ========== CALLING llama_decode with batch->n_tokens=%d ==========", batch->n_tokens);
    
    // 添加详细的batch参数验证和调试信息
    LOGi("[DEBUG] ========== BATCH PARAMETER VALIDATION ==========");
    LOGi("[DEBUG] batch pointer: %p", batch);
    LOGi("[DEBUG] batch->n_tokens: %d", batch->n_tokens);
    LOGi("[DEBUG] batch->token: %p", batch->token);
    LOGi("[DEBUG] batch->pos: %p", batch->pos);
    LOGi("[DEBUG] batch->n_seq_id: %p", batch->n_seq_id);
    LOGi("[DEBUG] batch->seq_id: %p", batch->seq_id);
    LOGi("[DEBUG] batch->logits: %p", batch->logits);
    
    // 添加内存连续性检查
    LOGi("[TRACE_POINT_3] Pre-llama_decode memory contiguity check");
    if (batch->token) {
        LOGi("[CONTIGUITY_CHECK] token array: start=%p, size=%zu bytes", 
             batch->token, sizeof(llama_token) * batch->n_tokens);
    }
    if (batch->pos) {
        LOGi("[CONTIGUITY_CHECK] pos array: start=%p, size=%zu bytes", 
             batch->pos, sizeof(llama_pos) * batch->n_tokens);
    }
    if (batch->seq_id) {
        LOGi("[CONTIGUITY_CHECK] seq_id pointer array: start=%p, size=%zu bytes", 
             batch->seq_id, sizeof(llama_seq_id*) * batch->n_tokens);
        for (int i = 0; i < batch->n_tokens && i < 3; ++i) {
            if (batch->seq_id[i]) {
                LOGi("[CONTIGUITY_CHECK] seq_id[%d] sub-array: start=%p", i, batch->seq_id[i]);
            }
        }
    }
    
    // 检查context参数
    LOGi("[DEBUG] ========== CONTEXT PARAMETER VALIDATION ==========");
    LOGi("[DEBUG] context pointer: %p", context);
    if (context) {
        LOGi("[DEBUG] context n_ctx: %d", llama_n_ctx(context));
        LOGi("[DEBUG] context n_batch: %d", llama_n_batch(context));
        const auto model = llama_get_model(context);
        if (model) {
            LOGi("[DEBUG] model pointer: %p", model);
            
            // 添加模型配置参数打印
            LOGi("[MODEL_CONFIG] ========== MODEL CONFIGURATION ===========");
            LOGi("[MODEL_CONFIG] n_ctx_train: %d", llama_model_n_ctx_train(model));
            LOGi("[MODEL_CONFIG] n_embd: %d", llama_model_n_embd(model));
            LOGi("[MODEL_CONFIG] n_layer: %d", llama_model_n_layer(model));
            LOGi("[MODEL_CONFIG] n_head: %d", llama_model_n_head(model));
            const auto vocab = llama_model_get_vocab(model);
            if (vocab) {
                LOGi("[MODEL_CONFIG] vocab_size: %d", llama_vocab_n_tokens(vocab));
            }
            LOGi("[MODEL_CONFIG] rope_freq_scale: %f", llama_model_rope_freq_scale_train(model));
            LOGi("[MODEL_CONFIG] =============================================");
        }
    }
    
    // 验证batch中的每个token数据
    LOGi("[DEBUG] ========== TOKEN DATA VALIDATION ==========");
    for (int i = 0; i < batch->n_tokens && i < 5; ++i) {
        LOGi("[DEBUG] token[%d]: value=%d, pos=%d, seq_id[0]=%d, logits=%s", 
             i, batch->token[i], batch->pos[i], 
             batch->seq_id[i] ? batch->seq_id[i][0] : -999, 
             batch->logits[i] ? "true" : "false");
    }
    
    // 检查内存对齐和数据完整性
    LOGi("[DEBUG] ========== MEMORY ALIGNMENT CHECK ==========");
    LOGi("[DEBUG] batch struct size: %zu", sizeof(*batch));
    LOGi("[DEBUG] token array alignment: %p (mod 8 = %ld)", batch->token, (long)batch->token % 8);
    LOGi("[DEBUG] pos array alignment: %p (mod 8 = %ld)", batch->pos, (long)batch->pos % 8);
    
    LOGi("[TRACE_POINT_4] About to call llama_decode");
    int decode_result = llama_decode(context, *batch);
    LOGi("[TRACE_POINT_5] llama_decode returned: %d", decode_result);
    LOGi("[DEBUG] ========== llama_decode returned: %d ==========", decode_result);
    
    if (decode_result != 0) {
        LOGe("llama_decode() failed with code: %d", decode_result);
        LOGi("[TRACE_POINT_6] llama_decode failed, cleaning up");
        env->ReleaseStringUTFChars(jtext, text);
        return -1;
    }
    
    LOGi("[TRACE_POINT_7] llama_decode completed successfully");
    LOGi("[DEBUG] llama_decode completed successfully");

    env->ReleaseStringUTFChars(jtext, text);
    
    LOGi("[DEBUG] ========== completion_init END: returning %d tokens ==========", batch->n_tokens);

    return batch->n_tokens;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_completion_1loop(
        JNIEnv * env,
        jobject,
        jlong context_pointer,
        jlong batch_pointer,
        jlong sampler_pointer,
        jint n_len,
        jobject intvar_ncur
) {
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto batch   = reinterpret_cast<llama_batch   *>(batch_pointer);
    const auto sampler = reinterpret_cast<llama_sampler *>(sampler_pointer);
    const auto model = llama_get_model(context);
    const auto vocab = llama_model_get_vocab(model);

    if (!la_int_var) la_int_var = env->GetObjectClass(intvar_ncur);
    if (!la_int_var_value) la_int_var_value = env->GetFieldID(la_int_var, "value", "I");
    if (!la_int_var_inc) la_int_var_inc = env->GetMethodID(la_int_var, "inc", "()V");

    // sample the most likely token
    const auto new_token_id = llama_sampler_sample(sampler, context, -1);

    const auto n_cur = env->GetIntField(intvar_ncur, la_int_var_value);
    if (llama_vocab_is_eog(vocab, new_token_id) || n_cur == n_len) {
        return nullptr;
    }

    auto new_token_chars = common_token_to_piece(context, new_token_id);
    cached_token_chars += new_token_chars;

    jstring new_token = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        new_token = env->NewStringUTF(cached_token_chars.c_str());
        LOGi("cached: %s, new_token_chars: `%s`, id: %d", cached_token_chars.c_str(), new_token_chars.c_str(), new_token_id);
        cached_token_chars.clear();
    } else {
        new_token = env->NewStringUTF("");
    }

    common_batch_clear(*batch);
    common_batch_add(*batch, new_token_id, n_cur, { 0 }, true);
    
    LOGi("[DEBUG] completion_loop: adding token_id=%d at pos=%d, batch->n_tokens=%d", new_token_id, n_cur, batch->n_tokens);

    env->CallVoidMethod(intvar_ncur, la_int_var_inc);

    if (llama_decode(context, *batch) != 0) {
        LOGe("llama_decode() failed in completion_loop");
        return nullptr;
    }
    
    LOGi("[DEBUG] completion_loop: llama_decode completed successfully");

    return new_token;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_kv_1cache_1clear(JNIEnv *, jobject, jlong context) {
    llama_kv_self_clear(reinterpret_cast<llama_context *>(context));
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_starlocalrag_llamacpp_LlamaCppInference_new_1sampler_1with_1params(
        JNIEnv *env,
        jobject,
        jfloat temperature,
        jfloat top_p,
        jint top_k
) {
    LOGi("[SAMPLER_CONFIG] ========== CREATING SAMPLER WITH PARAMS ==========");
    LOGi("[SAMPLER_CONFIG] Input parameters: temperature=%.6f, top_p=%.6f, top_k=%d", temperature, top_p, top_k);
    LOGi("[SAMPLER_CONFIG] LLAMA_DEFAULT_SEED value: 0x%08X (%u)", LLAMA_DEFAULT_SEED, LLAMA_DEFAULT_SEED);
    
    // 创建采样器参数
    auto sparams = llama_sampler_chain_default_params();
    LOGi("[SAMPLER_CONFIG] Default sampler chain params created");
    
    // 创建采样器链
    llama_sampler * sampler = llama_sampler_chain_init(sparams);
    LOGi("[SAMPLER_CONFIG] Sampler chain initialized: %p", sampler);
    
    // 添加top-k采样器
    if (top_k > 0) {
        auto top_k_sampler = llama_sampler_init_top_k(top_k);
        llama_sampler_chain_add(sampler, top_k_sampler);
        LOGi("[SAMPLER_CONFIG] Added top-k sampler: k=%d, sampler=%p", top_k, top_k_sampler);
    } else {
        LOGi("[SAMPLER_CONFIG] Skipped top-k sampler (k=%d <= 0)", top_k);
    }
    
    // 添加top-p采样器
    if (top_p > 0.0f && top_p < 1.0f) {
        auto top_p_sampler = llama_sampler_init_top_p(top_p, 1);
        llama_sampler_chain_add(sampler, top_p_sampler);
        LOGi("[SAMPLER_CONFIG] Added top-p sampler: p=%.6f, sampler=%p", top_p, top_p_sampler);
    } else {
        LOGi("[SAMPLER_CONFIG] Skipped top-p sampler (p=%.6f not in (0,1))", top_p);
    }
    
    // 添加温度采样器
    if (temperature > 0.0f) {
        auto temp_sampler = llama_sampler_init_temp(temperature);
        llama_sampler_chain_add(sampler, temp_sampler);
        LOGi("[SAMPLER_CONFIG] Added temperature sampler: temp=%.6f, sampler=%p", temperature, temp_sampler);
    } else {
        LOGi("[SAMPLER_CONFIG] Skipped temperature sampler (temp=%.6f <= 0)", temperature);
    }
    
    // 添加分布采样器 - 使用LLAMA_DEFAULT_SEED获得随机种子
    auto dist_sampler = llama_sampler_init_dist(LLAMA_DEFAULT_SEED);
    llama_sampler_chain_add(sampler, dist_sampler);
    LOGi("[SAMPLER_CONFIG] Added distribution sampler: seed=0x%08X, sampler=%p", LLAMA_DEFAULT_SEED, dist_sampler);
    
    if (!sampler) {
        LOGe("[SAMPLER_CONFIG] ERROR: new_sampler_with_params() failed - sampler is null");
        return 0;
    }
    
    LOGi("[SAMPLER_CONFIG] Successfully created sampler chain: %p", sampler);
    LOGi("[SAMPLER_CONFIG] Final parameters: temp=%.6f, top_p=%.6f, top_k=%d", temperature, top_p, top_k);
    LOGi("[SAMPLER_CONFIG] ======================================================");
    return reinterpret_cast<jlong>(sampler);
}
