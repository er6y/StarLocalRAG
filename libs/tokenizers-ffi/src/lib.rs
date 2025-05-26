use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use tokenizers::Tokenizer;
use tokenizers::models::{bpe::BPE, wordpiece::WordPiece};
use jni::JNIEnv;
use jni::objects::{JClass, JString};
use jni::sys::{jlong, jstring};
// use serde_json::Value; // 未使用的导入

// 创建分词器 - C API
#[no_mangle]
pub extern "C" fn create_tokenizer(model_type: *const c_char) -> *mut c_void {
    let model_type_str = unsafe {
        if model_type.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(model_type).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    
    // 由于没有 from_pretrained 方法，我们需要创建一个基本的分词器
    // 在实际使用中，应该从文件加载预训练的分词器
    let tokenizer = match model_type_str {
        "bpe" => {
            // 创建一个基本的 BPE 分词器
            let bpe = BPE::builder()
                .unk_token("[UNK]".into())
                .build()
                .unwrap();
            Tokenizer::new(bpe)
        },
        "wordpiece" => {
            // 创建一个基本的 WordPiece 分词器
            let wordpiece = WordPiece::builder()
                .unk_token("[UNK]".into())
                .build()
                .unwrap();
            Tokenizer::new(wordpiece)
        },
        _ => return std::ptr::null_mut(),
    };
    
    // 将 tokenizer 转换为指针并返回
    Box::into_raw(Box::new(tokenizer)) as *mut c_void
}

// 从文件加载分词器 - C API
#[no_mangle]
pub extern "C" fn load_tokenizer_from_file(path: *const c_char) -> *mut c_void {
    let path_str = unsafe {
        if path.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    
    let tokenizer = match Tokenizer::from_file(path_str) {
        Ok(t) => t,
        Err(_) => return std::ptr::null_mut(),
    };
    
    Box::into_raw(Box::new(tokenizer)) as *mut c_void
}

// 分词 - C API
#[no_mangle]
pub extern "C" fn tokenize(tokenizer_ptr: *mut c_void, text: *const c_char) -> *mut c_char {
    if tokenizer_ptr.is_null() || text.is_null() {
        return std::ptr::null_mut();
    }
    
    let tokenizer = unsafe { &mut *(tokenizer_ptr as *mut Tokenizer) };
    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    
    let encoding = match tokenizer.encode(text_str, false) {
        Ok(e) => e,
        Err(_) => return std::ptr::null_mut(),
    };
    
    let json = match serde_json::to_string(&encoding) {
        Ok(j) => j,
        Err(_) => return std::ptr::null_mut(),
    };
    
    match CString::new(json) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// 应用聊天模板功能已移至Java层面实现

// 获取分词器配置 - C API
#[no_mangle]
pub extern "C" fn get_tokenizer_config(tokenizer_ptr: *mut c_void) -> *mut c_char {
    if tokenizer_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    let tokenizer = unsafe { &*(tokenizer_ptr as *mut Tokenizer) };
    
    // 创建配置对象
    let mut config = serde_json::Map::new();
    
    // 获取词汇表
    let vocab = tokenizer.get_vocab(true);
    
    // 添加词汇表大小
    config.insert("vocab_size".to_string(), serde_json::Value::Number(vocab.len().into()));
    
    // 添加词汇表
    let mut vocab_json = serde_json::Map::new();
    for (token, id) in vocab.iter() {
        vocab_json.insert(token.clone(), serde_json::Value::Number((*id).into()));
    }
    config.insert("vocab".to_string(), serde_json::Value::Object(vocab_json));
    
    // 添加所有特殊 token
    let mut added_tokens = Vec::new();
    for (token, id) in vocab.iter() {
        // 检测可能的特殊 token
        if token.starts_with("<") || token.ends_with(">") ||
           token.starts_with("[") || token.ends_with("]") ||
           token == "user" || token == "assistant" || token == "system" {
            
            let mut token_obj = serde_json::Map::new();
            token_obj.insert("id".to_string(), serde_json::Value::Number((*id).into()));
            token_obj.insert("content".to_string(), serde_json::Value::String(token.clone()));
            token_obj.insert("special".to_string(), serde_json::Value::Bool(true));
            
            added_tokens.push(serde_json::Value::Object(token_obj));
        }
    }
    
    if !added_tokens.is_empty() {
        config.insert("added_tokens".to_string(), serde_json::Value::Array(added_tokens));
    }
    
    // 模型类型检测已移至Java层实现
    
    // 尝试获取模型类型
    // 这里我们无法直接获取模型类型，但可以根据词汇表特点推断
    
    // 将配置转换为JSON字符串
    let json = match serde_json::to_string(&config) {
        Ok(j) => j,
        Err(_) => return std::ptr::null_mut(),
    };
    
    match CString::new(json) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// 释放字符串 - C API
#[no_mangle]
pub extern "C" fn free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)) };
    }
}

// 解码token ID为文本 - C API
#[no_mangle]
pub extern "C" fn decode(tokenizer_ptr: *mut c_void, ids_json: *const c_char) -> *mut c_char {
    if tokenizer_ptr.is_null() || ids_json.is_null() {
        return std::ptr::null_mut();
    }
    
    let tokenizer = unsafe { &*(tokenizer_ptr as *mut Tokenizer) };
    
    // 将C字符串转换为Rust字符串
    let ids_str = match unsafe { CStr::from_ptr(ids_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    
    // 解析JSON数组
    let ids: Vec<u32> = match serde_json::from_str(ids_str) {
        Ok(i) => i,
        Err(_) => return std::ptr::null_mut(),
    };
    
    // 解码token ID
    let decoded = match tokenizer.decode(&ids, false) {
        Ok(text) => text,
        Err(_) => return std::ptr::null_mut(),
    };
    
    // 将结果转换为C字符串
    match CString::new(decoded) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// 释放分词器 - C API
#[no_mangle]
pub extern "C" fn free_tokenizer(tokenizer_ptr: *mut c_void) {
    if !tokenizer_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(tokenizer_ptr as *mut Tokenizer);
        }
    }
}

// JNI 接口实现

#[no_mangle]
pub extern "C" fn Java_com_starlocalrag_tokenizers_TokenizerJNI_createTokenizer__Ljava_lang_String_2(
    env: JNIEnv,
    _class: JClass,
    model_type: JString,
) -> jlong {
    let model_type_str: String = match env.get_string(model_type) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    
    let c_model_type = match CString::new(model_type_str) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    
    create_tokenizer(c_model_type.as_ptr()) as jlong
}

#[no_mangle]
pub extern "C" fn Java_com_starlocalrag_tokenizers_TokenizerJNI_loadTokenizerFromFile__Ljava_lang_String_2(
    env: JNIEnv,
    _class: JClass,
    path: JString,
) -> jlong {
    let path_str: String = match env.get_string(path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    
    let c_path = match CString::new(path_str) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    
    load_tokenizer_from_file(c_path.as_ptr()) as jlong
}

#[no_mangle]
pub extern "C" fn Java_com_starlocalrag_tokenizers_TokenizerJNI_tokenize__JLjava_lang_String_2(
    env: JNIEnv,
    _class: JClass,
    tokenizer_ptr: jlong,
    text: JString,
) -> jstring {
    // 从 JString 获取 Rust String，确保正确处理 UTF-8
    let text_str: String = match env.get_string(text) {
        Ok(s) => s.into(),
        Err(e) => {
            eprintln!("Error converting JString to Rust String: {:?}", e);
            return std::ptr::null_mut();
        }
    };
    
    // 将 Rust String 转换为 C 字符串
    let c_text = match CString::new(text_str) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error creating CString: {:?}", e);
            return std::ptr::null_mut();
        }
    };
    
    // 调用 C 函数进行分词
    let result = tokenize(tokenizer_ptr as *mut c_void, c_text.as_ptr());
    
    if result.is_null() {
        eprintln!("Tokenization returned null result");
        return std::ptr::null_mut();
    }
    
    // 将 C 字符串转换回 Rust 字符串，然后转换为 Java 字符串
    let result_str = unsafe { CStr::from_ptr(result) };
    let result_rust_str = match result_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error converting CStr to Rust str: {:?}", e);
            free_string(result);
            return std::ptr::null_mut();
        }
    };
    
    // 创建 Java 字符串
    let java_string = match env.new_string(result_rust_str) {
        Ok(s) => s.into_inner(),
        Err(e) => {
            eprintln!("Error creating Java string: {:?}", e);
            free_string(result);
            std::ptr::null_mut()
        }
    };
    
    // 释放 C 字符串
    free_string(result);
    
    java_string
}

// 应用聊天模板功能已移至Java层面实现

#[no_mangle]
pub extern "C" fn Java_com_starlocalrag_tokenizers_TokenizerJNI_getTokenizerConfig__J(
    env: JNIEnv,
    _class: JClass,
    tokenizer_ptr: jlong,
) -> jstring {
    // 调用 C 函数获取分词器配置
    let result = get_tokenizer_config(tokenizer_ptr as *mut c_void);
    
    if result.is_null() {
        eprintln!("Get tokenizer config returned null result");
        return std::ptr::null_mut();
    }
    
    // 将 C 字符串转换回 Rust 字符串，然后转换为 Java 字符串
    let result_str = unsafe { CStr::from_ptr(result) };
    let result_rust_str = match result_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error converting CStr to Rust str: {:?}", e);
            free_string(result);
            return std::ptr::null_mut();
        }
    };
    
    // 创建 Java 字符串
    let java_string = match env.new_string(result_rust_str) {
        Ok(s) => s.into_inner(),
        Err(e) => {
            eprintln!("Error creating Java string: {:?}", e);
            free_string(result);
            std::ptr::null_mut()
        }
    };
    
    // 释放 C 字符串
    free_string(result);
    
    java_string
}

#[no_mangle]
pub extern "C" fn Java_com_starlocalrag_tokenizers_TokenizerJNI_decode__JLjava_lang_String_2(
    env: JNIEnv,
    _class: JClass,
    tokenizer_ptr: jlong,
    ids_json: JString,
) -> jstring {
    // 从 JString 获取 Rust String，确保正确处理 UTF-8
    let ids_str: String = match env.get_string(ids_json) {
        Ok(s) => s.into(),
        Err(e) => {
            eprintln!("Error converting JString to Rust String: {:?}", e);
            return std::ptr::null_mut();
        }
    };
    
    // 将 Rust String 转换为 C 字符串
    let c_ids = match CString::new(ids_str) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error converting Rust String to CString: {:?}", e);
            return std::ptr::null_mut();
        }
    };
    
    // 调用 C 函数进行解码
    let result = decode(tokenizer_ptr as *mut c_void, c_ids.as_ptr());
    
    if result.is_null() {
        eprintln!("Decoding returned null result");
        return std::ptr::null_mut();
    }
    
    // 将 C 字符串转换回 Rust 字符串，然后转换为 Java 字符串
    let result_str = unsafe { CStr::from_ptr(result) };
    let result_rust_str = match result_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error converting CStr to Rust str: {:?}", e);
            free_string(result);
            return std::ptr::null_mut();
        }
    };
    
    // 创建 Java 字符串
    let java_string = match env.new_string(result_rust_str) {
        Ok(s) => s.into_inner(),
        Err(e) => {
            eprintln!("Error creating Java string: {:?}", e);
            free_string(result);
            std::ptr::null_mut()
        }
    };
    
    // 释放 C 字符串
    free_string(result);
    
    java_string
}

#[no_mangle]
pub extern "C" fn Java_com_starlocalrag_tokenizers_TokenizerJNI_freeTokenizer__J(
    _env: JNIEnv,
    _class: JClass,
    tokenizer_ptr: jlong,
) {
    free_tokenizer(tokenizer_ptr as *mut c_void);
}
