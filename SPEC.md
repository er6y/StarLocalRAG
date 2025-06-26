# StarLocalRAG 技术规格文档

## 1. 项目概述

### 1.1 项目简介

StarLocalRAG 是一个基于 Android 平台的本地 RAG（Retrieval-Augmented Generation）应用，支持完全离线的知识库构建、文档检索和智能问答功能。

### 1.2 核心特性

- **完全本地化**：所有功能在设备本地运行，保护用户隐私
- **多模型支持**：集成 ONNX Runtime 和 LlamaCpp 推理引擎
- **智能检索**：支持向量检索和重排模型优化
- **多格式文档**：支持 PDF、Word、Excel、文本等多种文档格式
- **流式推理**：实时显示 AI 生成内容，支持推理中断

### 1.3 技术架构

- **前端**：Android 原生应用（Java）
- **推理引擎**：ONNX Runtime 1.21.0、LlamaCpp JNI
- **分词器**：Rust 实现的高性能分词器
- **向量数据库**：SQLite 本地向量存储
- **文档处理**：多格式文档解析和文本分块

## 2. 功能模块

### 2.1 RAG 问答系统

#### 2.1.1 核心功能

- **智能问答**：基于知识库的检索增强生成
- **模型选择**：支持本地 LLM 和 API 模式
- **参数配置**：检索数、重排数、思考模式等可配置
- **结果管理**：支持复制、转笔记、文本缩放等操作

#### 2.1.2 界面优化

- **紧凑布局**：横向排列相关功能组件
- **配置记忆**：自动保存用户选择的模型和参数
- **思考模式**：支持 `/no_think` 指令控制推理模式

#### 2.1.3 推理参数优先级

**参数来源优先级**（从高到低）：
1. 模型目录配置文件（`params` 或 `generation_config.json`）
2. 用户手动配置参数
3. 系统默认参数

### 2.2 知识库构建

#### 2.2.1 构建流程

```
选择知识库和模型 → 添加文档 → 文档解析 → 文本分块 → 向量化 → 存储到数据库
```

#### 2.2.2 模型匹配检查

- **覆盖模式**：删除现有数据，使用新模型重建
- **追加模式**：检查向量维度匹配，不匹配时提示用户
- **智能验证**：自动检测模型兼容性

#### 2.2.3 重排模型集成

- **架构优化**：统一异步处理模式，避免嵌套线程池
- **输入格式**：使用 `[Q] query [SEP] [D] document [SEP]` 标记格式
- **超时配置**：推理超时时间调整为 2 分钟

### 2.3 本地 LLM 推理

#### 2.3.1 推理引擎

**LlamaCpp 集成**：
- JNI 接口优化，修复方法名匹配问题
- 内存管理改进，修复官方实现的内存泄漏
- 批处理优化，支持动态 batch 大小

**ONNX Runtime**：
- 版本升级到 1.21.0
- GPU 加速支持和自动降级
- CPU 多线程优化

#### 2.3.2 推理功能

**流式输出**：
- 实时 token 生成显示
- `StreamingCallback` 接口处理
- 批量解码优化

**推理中断**：
- 线程安全的停止标志机制
- Java 和 JNI 层状态同步
- 即时响应用户中断指令

**性能统计**：
- Token 生成速度监控
- 内存使用统计
- 推理配置信息展示

**推理结束token识别优化**：
- 添加详细的debug日志，用于排查推理结束token识别问题
- 特别针对Qwen3等模型的特殊token识别进行优化
- 在JNI层添加token采样、特殊token检测和推理结束判断的详细日志
- 通过`llama_vocab_is_eog`函数准确判断生成结束token，依赖模型自身的EOG判断逻辑
- 移除硬编码的结束token列表，避免过度敏感的误判问题
- 提供推理结束条件的详细诊断信息，包括EOG检测和长度限制

### 2.4 模型下载管理

#### 2.4.1 下载功能

- **分类管理**：嵌入模型、重排模型、LLM 分类展示
- **批量下载**：支持多模型并发下载
- **进度监控**：实时进度显示和状态更新
- **智能检测**：Wi-Fi 连接检查和目录冲突处理

#### 2.4.2 下载优化

- **中断逻辑**：点击停止立即中断，不尝试备用地址
- **电源管理**：自动获取唤醒锁和 Wi-Fi 锁
- **进度显示**：优化的百分比显示格式

## 3. 技术实现

### 3.1 分词器系统

#### 3.1.1 Rust 分词器

**架构设计**：
- Rust 核心分词逻辑
- 直接 JNI 接口（消除 FFI 中间层）
- Java 封装层

**性能优化**：
- 多架构支持（ARM64、ARMv7、x86_64）
- 内存优化策略
- 高效缓存机制

#### 3.1.2 本地 LLM 分词

- 专用 `LocalLLMTokenizer`，与 RAG 分词器隔离
- 支持特殊 token 处理（如 `<|im_start|>`、`<|im_end|>`）
- 完善对话模板格式化

### 3.2 构建系统

#### 3.2.1 Android 构建优化

**CMake 配置**：
- 静态库改为共享库
- 禁用 CURL 依赖
- 强制 O3 编译优化

**兼容性修复**：
- Android POSIX 兼容性
- ARM NEON FP16 支持
- x86 模拟器兼容性

#### 3.2.2 F16C 指令集处理

**2025年1月重大更新**：
- Android 平台全面禁用 F16C
- 优先保证模拟器环境稳定性
- 统一编译器标志设置

#### 3.2.3 Tokenizers-JNI 构建修复

**问题诊断**：
- `cargo ndk` 命令参数格式错误，导致自动构建失败
- 缺少 x86 和 x86_64 架构的编译任务
- 手动编译成功但 Gradle 构建流程未正确执行

**修复实现**：
```gradle
// 修正 cargo ndk 命令参数格式
commandLine 'cmd', '/c', 'cargo', 'ndk', '--target', 'aarch64-linux-android', '--', 'build', '--release'
// 原错误格式：'-t', 'arm64-v8a'

// 添加完整架构支持
task compileRustJNIAndroidX86(type: Exec) {
    commandLine 'cmd', '/c', 'cargo', 'ndk', '--target', 'i686-linux-android', '--', 'build', '--release'
}

task compileRustJNIAndroidX86_64(type: Exec) {
    commandLine 'cmd', '/c', 'cargo', 'ndk', '--target', 'x86_64-linux-android', '--', 'build', '--release'
}
```

**架构支持完善**：
- ARM64 (`aarch64-linux-android`)
- ARMv7 (`armv7-linux-androideabi`)
- x86 (`i686-linux-android`)
- x86_64 (`x86_64-linux-android`)

**构建流程优化**：
- 统一编译任务命名规范
- 完善复制任务依赖关系
- 确保 `preBuild` 正确依赖所有架构的复制任务

**2025年1月构建流程重构（方案A实施）**：
- **直接输出策略**：使用 `CARGO_TARGET_DIR` 环境变量直接将 Rust 编译输出指向 `app/src/main/jniLibs` 目录
- **消除复制步骤**：删除所有 `copyRustLibraries*` 任务，简化构建流程
- **统一JNI库管理**：与 `llamacpp-jni` 保持一致的输出策略，所有 `.so` 文件统一输出到 `app/src/main/jniLibs`
- **依赖关系优化**：`preBuild` 任务直接依赖编译任务而非复制任务
- **重复文件清理**：清理现有重复的 `.so` 文件，避免构建警告

```gradle
// 方案A核心实现：直接输出到目标目录
task compileRustJNIAndroidArm64(type: Exec) {
    environment 'CARGO_TARGET_DIR', file('../../app/src/main/jniLibs/arm64-v8a').absolutePath
    commandLine 'cmd', '/c', 'cargo', 'ndk', '--target', 'aarch64-linux-android', '--', 'build', '--release'
}
```

### 3.3 内存管理

#### 3.3.1 JNI 层优化

- 修复 `free_batch` 函数内存泄漏
- 添加全局状态管理和互斥锁
- 完善参数验证和边界检查

#### 3.3.2 JVM 内存配置

- 启用大内存模式（`android:largeHeap="true"`）
- 增加 Gradle 守护进程内存
- 实现内存监控系统

#### 3.3.3 KV 缓存优化

- 修复张量类型转换问题
- 自适应缓存大小调整
- 根据内存状况动态禁用缓存

### 3.4 错误处理与调试

#### 3.4.1 SIGSEGV 错误修复

**LlamaCpp JNI 优化**：
- `JNI_OnLoad` 初始化改进
- Context 创建参数验证
- 批处理大小智能调整
- 分词函数两阶段调用

**x86 模拟器优化**：
- 禁用可能导致崩溃的特性
- 更保守的批处理策略
- 增强内存安全检查

#### 3.4.2 GGML 矩阵计算修复

- Logits 指针修复
- 批处理验证增强
- 源码级调试支持
- Tensor 步长计算修复

#### 3.4.3 调试系统

- 跨平台日志支持
- 条件编译控制
- 详细错误信息记录
- 性能监控集成

## 4. 配置管理

### 4.1 参数配置系统

#### 4.1.1 配置文件支持

- `params` 文件（键值对格式）
- `generation_config.json` 文件（JSON 格式）
- 自动格式检测和解析

#### 4.1.2 配置优先级

```java
public class ModelParamsReader {
    public static InferenceParams readInferenceParams(String modelDirPath) {
        // 1. 尝试读取 params 文件
        File paramsFile = new File(modelDir, "params");
        if (paramsFile.exists()) {
            return readFromParamsFile(paramsFile);
        }
        
        // 2. 尝试读取 generation_config.json 文件
        File configFile = new File(modelDir, "generation_config.json");
        if (configFile.exists()) {
            return readFromJsonFile(configFile);
        }
        
        return null; // 使用默认参数
    }
}
```

### 4.2 用户配置管理

#### 4.2.1 持久化存储

- 使用 `SharedPreferences` 存储用户配置
- 支持配置导入导出
- 自动配置备份和恢复

#### 4.2.2 配置项管理

- 模型选择记忆
- 推理参数设置
- 界面状态保存
- 重排模型路径配置

## 5. 性能优化

### 5.1 推理性能

#### 5.1.1 日志优化

减少高频调试日志输出：
- `ggml-cpu.c` 张量操作日志
- `llama-graph.cpp` attention 计算日志
- `ops.cpp` transpose 操作日志
- `ggml.c` 张量创建日志

#### 5.1.2 批处理优化

- 动态 batch 大小处理
- 智能批处理策略
- 内存使用优化

### 5.2 GPU 加速

#### 5.2.1 HarmonyOS 适配

- HarmonyOS GPU 特性优化
- Mali GPU 性能优化
- OpenGL ES 计算着色器支持

#### 5.2.2 错误处理

- GPU 可用性检测
- 自动降级策略
- 详细错误信息提供

### 5.3 内存优化

#### 5.3.1 资源管理

- 模型预分配策略
- 资源复用机制
- 自动资源清理

#### 5.3.2 监控系统

- 实时内存使用监控
- 性能趋势分析
- 资源使用统计

## 6. 开发规范

### 6.1 代码质量

#### 6.1.1 编译优化

- 特殊字符处理
- 类型转换优化
- 接口实现完整性检查

#### 6.1.2 架构优化

- 死代码清理
- 依赖管理优化
- 统一网络请求架构

### 6.2 最佳实践

#### 6.2.1 内存安全

- RAII 原则应用
- 空指针检查
- 资源自动释放

#### 6.2.2 线程安全

- 原子操作使用
- 互斥锁保护
- 状态同步机制

#### 6.2.3 错误处理

- 详细错误信息
- 用户友好提示
- 自动恢复机制

## 7. 国际化与多语言支持

### 7.1 硬编码中文逻辑判断重构

#### 7.1.1 问题背景

**严重性评估**：应用存在大量硬编码中文逻辑判断，完全阻碍国际化进程

**影响范围**：
- 构建知识库模块：模型状态判断、进度状态解析
- RAG问答模块：API选择、模型选择、知识库选择
- 辅助功能：配置管理、模型下载、日志系统

#### 7.1.2 架构重构方案

**核心组件设计**：

```java
// 状态常量统一管理
public class AppConstants {
    // 模型状态常量
    public static final String MODEL_STATE_LOADING = "loading";
    public static final String MODEL_STATE_NO_AVAILABLE = "no_available";
    public static final String MODEL_STATE_FETCH_FAILED = "fetch_failed";
    
    // 知识库状态常量
    public static final String KB_STATE_LOADING = "kb_loading";
    public static final String KB_STATE_NONE = "kb_none";
    public static final String KB_STATE_PLEASE_CREATE = "kb_please_create";
    
    // 进度状态常量
    public static final String PROGRESS_STATE_EXTRACTING_TEXT = "extracting_text";
    public static final String PROGRESS_STATE_GENERATING_VECTORS = "generating_vectors";
}

// 状态显示管理器
public class StateDisplayManager {
    public String getModelStateDisplayText(String stateKey) {
        switch (stateKey) {
            case AppConstants.MODEL_STATE_LOADING:
                return context.getString(R.string.model_state_loading);
            // ... 其他状态映射
        }
    }
    
    public boolean isValidModelState(String stateKey) {
        return !AppConstants.MODEL_STATE_LOADING.equals(stateKey) &&
               !AppConstants.MODEL_STATE_NO_AVAILABLE.equals(stateKey);
    }
}

// 状态感知的Spinner适配器
public class StateAwareSpinnerAdapter extends ArrayAdapter<String> {
    private List<String> stateKeys;
    private StateDisplayManager stateDisplayManager;
    
    public String getStateKeyAtPosition(int position) {
        return stateKeys.get(position);
    }
}
```

#### 7.1.3 重构实施策略

**分阶段实施**：
1. **基础架构建设**（2-3天）：创建状态常量类、显示管理器、状态感知适配器
2. **核心模块重构**（4-5天）：重构BuildKnowledgeBaseFragment和RagQaFragment
3. **辅助功能重构**（3-4天）：重构配置管理、模型下载、日志系统
4. **多语言支持完善**（2-3天）：完善字符串资源，添加英文翻译
5. **测试验证与优化**（3-4天）：功能测试、兼容性测试、性能测试

**风险控制措施**：
- 立即停止多语言功能发布，直到重构完成
- 建立配置文件平滑迁移机制
- 使用功能开关控制新旧逻辑切换
- 为每个重构阶段准备快速回滚方案

#### 7.1.4 技术实施要点

**状态管理统一化**：
- 所有状态使用常量定义，避免魔法字符串
- 状态与显示分离，支持动态语言切换
- 状态验证逻辑集中管理，避免重复代码

**适配器重构策略**：
- 使用状态感知适配器，自动处理显示文本转换
- 保持适配器接口兼容性，减少调用方修改
- 支持动态数据更新，适应状态变化

**资源管理优化**：
- 字符串资源分类管理，建立命名规范
- 支持参数化字符串，提高灵活性
- 建立多语言资源文件（中文、英文）

### 7.2 多语言架构设计

#### 7.2.1 按钮文本国际化修复

**问题描述**：
在知识库构建界面中，"CANCEL"按钮在中文界面下仍显示英文，未能正确根据语言设置显示为"取消"。

**问题分析**：
1. `StateDisplayManager.getButtonDisplayText()` 方法调用 `getDisplayText("button", key)`
2. `getDisplayText()` 方法的 switch 语句中缺少 "button" 类型的处理
3. 导致按钮文本直接返回原始常量值而非本地化字符串

**修复实现**：

1. **添加 `getButtonDisplay` 方法**：
```java
public String getButtonDisplay(String buttonKey) {
    switch (buttonKey) {
        case AppConstants.BUTTON_TEXT_OK:
            return context.getString(R.string.button_ok);
        case AppConstants.BUTTON_TEXT_CANCEL:
            return context.getString(R.string.button_cancel);
        case AppConstants.BUTTON_TEXT_CREATE_KB:
            return context.getString(R.string.button_create_kb);
        case AppConstants.BUTTON_TEXT_CONTINUE:
            return context.getString(R.string.button_continue);
        case AppConstants.BUTTON_TEXT_NEW_KB:
            return context.getString(R.string.button_new_kb);
        default:
            return buttonKey;
    }
}
```

2. **更新 `getDisplayText` 方法**：
在 switch 语句中添加 "button" 类型处理：
```java
case "button":
    return getButtonDisplay(state);
```

**修复效果**：
- 中文界面下，"CANCEL" 按钮正确显示为 "取消"
- 英文界面下，按钮显示为 "Cancel"
- 其他按钮文本（确定、继续、创建知识库等）也能正确本地化

#### 7.2.2 字符串资源管理

**资源文件结构**：
```
res/
├── values/
│   └── strings.xml          # 默认中文资源
├── values-en/
│   └── strings.xml          # 英文资源
└── values-zh-rCN/
    └── strings.xml          # 简体中文资源
```

**命名规范**：
- 模型状态：`model_state_*`
- 知识库状态：`kb_state_*`
- 进度状态：`progress_*`
- 错误信息：`error_*`
- 用户提示：`prompt_*`

#### 7.2.2 动态语言切换

**实现机制**：
- 使用Android系统语言设置
- 支持应用内语言切换（未来扩展）
- 状态显示管理器自动适配当前语言

**兼容性保障**：
- 现有配置文件自动迁移
- 状态常量向后兼容
- 降级方案确保功能稳定

#### 7.2.3 验证消息国际化修复

**问题描述**：
- 知识库构建时验证错误消息未正确国际化
- `StateDisplayManager.getValidationDisplayText` 方法缺少 validation 类型处理
- 导致错误消息显示为常量键值而非本地化文本

**修复实现**：
```java
// StateDisplayManager 添加验证消息处理
public String getValidationDisplay(String validationKey) {
    switch (validationKey) {
        case AppConstants.VALIDATION_PLEASE_SELECT_VALID_EMBEDDING:
            return context.getString(R.string.validation_please_select_valid_embedding);
        case AppConstants.VALIDATION_PLEASE_SELECT_FILES:
            return context.getString(R.string.validation_please_select_files);
        case AppConstants.VALIDATION_KB_NAME_CANNOT_BE_EMPTY:
            return context.getString(R.string.validation_kb_name_cannot_be_empty);
        default:
            return context.getString(R.string.validation_unknown);
    }
}

// getDisplayText 方法添加 validation 类型支持
case "validation":
    return getValidationDisplay(state);
```

**资源文件完善**：
- 添加 `validation_unknown` 字符串资源
- 确保中英文资源文件同步
- 完善验证消息的多语言支持

#### 7.2.4 嵌入模型验证逻辑优化

**问题分析**：
- 嵌入模型验证逻辑过于简单，未考虑加载状态
- 缺少详细的调试日志，难以定位问题
- 验证条件不够严格，可能误判有效模型

**优化实现**：
```java
// 改进的验证逻辑
boolean isValidEmbedding = selectedPosition >= 0 && 
                          !embeddingModel.equals("加载中...") && 
                          !embeddingModel.equals("无可用模型") &&
                          !StateDisplayManager.isModelStatusDisplayText(requireContext(), embeddingModel);

// 添加详细日志
LogManager.logD(TAG, "当前选择的嵌入模型: " + embeddingModel + ", 位置: " + selectedPosition);
LogManager.logW(TAG, "验证失败：未选择有效的嵌入模型，当前选择: " + embeddingModel);
```

#### 7.2.5 日志国际化实现

**日志资源管理**：
- 日志消息统一使用字符串资源ID（`R.string.*`）
- 按模块分类管理日志资源（如 `config_*`, `model_*`, `kb_*`）
- 支持格式化参数的日志消息（如 `getString(R.string.log_format, arg1, arg2)`）

**日志工具链**：
- `LogManager` 提供统一的日志记录接口
- 各模块通过辅助方法 `getLogString` 将资源ID转换为本地化文本
- 避免在日志中使用硬编码文本，确保所有日志可国际化

**实现最佳实践**：
```java
// 日志常量定义
private static final int LOG_OPERATION_FAILED = R.string.operation_failed;

// 获取本地化日志文本
private static String getLogString(Context context, int resId) {
    return context != null ? context.getString(resId) : "";
}

// 日志记录调用
LogManager.logE(TAG, getLogString(context, LOG_OPERATION_FAILED), e);
```

**优化与注意事项**：
1.  **避免重复的字符串资源**：在添加新的字符串资源之前，必须先在整个项目中（特别是 `values*/strings.xml` 文件中）搜索，确认该资源键（name）尚未存在。重复的资源键会导致资源合并冲突，从而导致编译失败。推荐使用 Android Studio 的“Find in Files”功能进行全局搜索。
2.  **保持多语言资源同步**：对于每一种语言（如 `values-en`, `values-zh-rCN`），其 `strings.xml` 文件中定义的资源项应与默认的 `values/strings.xml` 文件保持一致。如果某个资源只在特定语言文件中存在，编译时会产生 `removing resource ... without required default value` 警告。虽然这不影响编译，但为了应用的健壮性和可维护性，应确保所有语言的资源文件都包含相同的资源键。
3.  **正确格式化带占位符的字符串**：当字符串资源包含多个占位符时（如 `%s`, `%d`），必须使用位置格式（如 `%1$s`, `%2$d`）来明确指定参数顺序，否则会导致编译错误和潜在的运行时异常。

## 8. 未来发展

### 8.1 功能扩展

- 更多模型格式支持
- 多模态模型集成
- 跨平台版本开发
- 更多语言支持（日语、韩语等）

### 8.2 性能提升

- NPU 硬件加速
- 分布式推理支持
- 模型压缩技术
- 状态管理性能优化

### 8.3 用户体验

- 智能参数调优
- 可视化调试工具
- 个性化配置推荐
- 用户自定义语言包

### 8.4 开发效率

- 代码生成工具（自动生成状态常量）
- 国际化检查工具（确保文本完整性）
- 自动化测试框架
- 监控和分析系统

---

*本文档记录 StarLocalRAG 项目的核心技术实现和最佳实践，持续更新以反映最新的技术演进。*
