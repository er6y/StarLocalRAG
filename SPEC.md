# StarLocalRAG 应用规格说明

## 1. 功能需求

### 1.1 应用概述

StarLocalRAG 是一个安卓本地 RAG（检索增强生成）应用，允许用户在设备本地构建知识库、进行问答交互和管理知识笔记，无需依赖云服务。应用支持多种文档格式，使用嵌入模型进行向量化，并通过本地数据库进行高效检索。

### 1.2 页面与功能

应用包含三个主要页面，通过底部导航栏进行切换：

#### 1.2.1 RAG问答页面 (RagQaFragment)

**功能：**
- 选择API服务器、API密钥和模型
- 选择知识库进行问答
- 设置系统提示词
- 输入用户问题并获取AI回答
- 支持停止生成
- 支持将AI回答转为笔记
- 支持文本缩放
- 支持复制回答内容
- 支持长按文本选择和上下文菜单操作（复制、全选、转笔记等）

#### 1.2.2 构建知识库页面 (BuildKnowledgeBaseFragment)

**功能：**
- 创建新知识库
- 选择现有知识库进行更新
- 选择嵌入模型
- 浏览并添加文件到知识库
- 处理多种文档格式（文本、PDF、Word、Excel等）
- 显示处理进度
- 支持取消处理任务
- 支持防锁屏模式

#### 1.2.3 知识笔记页面 (KnowledgeNoteFragment)

**功能：**
- 选择知识库
- 创建笔记标题和内容
- 将笔记添加到知识库
- 支持从RAG问答页面转移内容
- 支持文本缩放
- 自动选择或提示选择嵌入模型

## 2. 功能设计

### 2.1 整体架构

StarLocalRAG 应用采用模块化设计，主要包括以下核心组件：

1. **UI层**：Activity和Fragment
2. **业务逻辑层**：各种Manager和Handler类
3. **数据处理层**：文本处理、向量化和数据库操作
4. **工具类**：配置管理、文件操作等

### 2.2 核心流程

#### 2.2.1 知识库构建流程

```
用户选择知识库名称和嵌入模型
↓
用户选择文件
↓
BuildKnowledgeBaseFragment.processKnowledgeBase()
↓
DocumentParser 解析文档提取文本
↓
TextProcessor 进行文本分块
↓
EmbeddingModelHandler 生成文本嵌入向量
↓
SQLiteVectorDatabaseHandler 存储向量到数据库
```

#### 2.2.2 RAG问答流程

```
用户选择知识库和API设置
↓
用户输入问题
↓
RagQaFragment.executeRagQuery()
↓
EmbeddingModelHandler 生成问题的嵌入向量
↓
SQLiteVectorDatabaseHandler 检索相关文本块
↓
构建包含检索结果的提示词
↓
通过API发送到LLM获取回答
↓
显示回答结果
```

#### 2.2.3 知识笔记添加流程

```
用户输入笔记内容
↓
KnowledgeNoteFragment.addToKnowledgeBase()
↓
TextProcessor 进行文本分块
↓
EmbeddingModelHandler 生成文本嵌入向量
↓
SQLiteVectorDatabaseHandler 存储向量到数据库
```

#### 2.2.4 模型匹配检查

**设计要点：**
1. **覆盖模式**：
   - 直接使用用户选择的新模型
   - 删除现有知识库的所有相关文件（向量数据库和元数据）
   - 重新构建知识库

2. **追加模式**：
   - 检查现有知识库的向量维度和模型路径是否匹配
   - 如果维度不匹配：强制停止并提示用户使用覆盖模式或选择匹配维度的模型
   - 如果模型路径不匹配但维度匹配：弹出警告对话框，让用户选择是否继续
   - 用户可以选择继续使用新模型或取消操作

**实现细节：**
- 使用SQLiteVectorDatabaseHandler获取现有知识库的元数据
- 比较现有模型的路径和维度与新模型是否匹配
- 使用AlertDialog显示警告信息
- 正确处理用户选择继续或取消的操作流程

## 3. 优化与注意事项

### 3.1 向量维度处理

在RAG应用中，向量维度的一致性至关重要。应用现已实现以下机制确保向量维度匹配：

1. **维度获取**：
   - 优先从模型配置文件中获取向量维度
   - 如无法获取，使用默认值1024作为向量维度

2. **维度检查**：
   - 构建知识库时，在追加模式下检查现有知识库与当前模型的向量维度是否匹配
   - 如维度不匹配，在追加模式下会停止处理并提示用户使用覆盖模式或选择匹配维度的模型
   - 在覆盖模式下，会删除现有数据库文件，使用新模型的维度重建知识库

3. **查询时检查**：
   - RAG查询时会检查查询向量与知识库向量的维度是否匹配
   - 如维度不匹配，会向用户显示警告信息，建议重建知识库或使用匹配维度的模型

这种设计确保了向量操作的正确性，同时为用户提供了清晰的错误信息和解决方案。

### 3.2 编译与构建优化

1. **编译错误修复最佳实践**
   - **特殊字符处理**：移除代码中的特殊Unicode字符（如✓、✗等），这些字符可能导致编译器无法正确解析代码
   - **类型转换优化**：
     - 对于必要的未检查类型转换，使用`@SuppressWarnings("unchecked")`注解抑制警告
     - 移除冗余的类型转换，如当方法返回类型已经是目标类型时，避免不必要的强制转换
     - 确保数组维度声明中使用正确的数据类型，避免不必要的int强制转换
   - **接口实现完整性**：确保匿名类或实现类完整实现接口中的所有抽象方法
   - **编译配置优化**：
     - 在`build.gradle`中添加详细的编译选项，如`-Xlint:all`显示所有警告
     - 设置`-Xmaxerrs`和`-Xmaxwarns`参数控制错误和警告的显示数量
     - 配置编码为UTF-8，确保中文注释和字符串的正确处理
   - **渐进式错误修复**：按照编译器报告的错误优先级逐一修复，先解决语法错误，再处理警告
   - **代码质量保证**：定期运行完整构建检查，确保所有模块都能正确编译

### 3.3 性能优化

1. **内存管理**
   - 使用单例模式管理嵌入模型，避免重复加载
   - 实现模型自动卸载机制，长时间不使用时释放资源
   - 使用线程池和异步处理，避免阻塞主线程
   - 优化向量数据库查询性能，使用SQLite事务提高写入速度
   - 保留完整上下文信息，确保RAG查询效果最佳
   - 使用内存优化的ONNX会话选项，如内存模式优化
   - 根据系统CPU核心数动态调整线程数，平衡性能和资源使用
   - 实现内存监控系统，周期性记录应用和系统内存使用情况
   - 在模型加载、推理和卸载过程中记录内存状态，便于分析内存瓶颈
   - 开启ONNX Runtime详细日志，便于排查内存占用问题
   - 定期手动触发GC，及时回收不再使用的内存
   - 安全关闭所有张量和会话资源，避免内存泄漏
   - **JVM内存配置优化**：
     * AndroidManifest.xml中配置`android:largeHeap="true"`启用大内存模式
     * gradle.properties中配置`org.gradle.jvmargs=-Xmx4096m`增加Gradle守护进程内存
     * build.gradle中配置`dexOptions.javaMaxHeapSize "4g"`增加编译时内存
     * GlobalApplication中实现内存监控，实时显示JVM和系统内存状态
   - **KV缓存优化**：
     * 修复KV缓存张量类型转换问题，支持IntBuffer和FloatBuffer多种数据类型
     * 实现自适应缓存大小调整，根据内存状况动态禁用KV缓存
     * 添加详细的KV缓存创建日志，便于问题诊断
     * 解决cache_length张量创建时的LongBuffer类型错误
   
2. **LLM分词与推理优化**
   - 实现专用的LocalLLMTokenizer，与RAG知识库分词器完全隔离
   - 正确加载和处理Qwen3模型的特殊token（如<|im_start|>、<|im_end|>等）
   - 增强分词算法，支持特殊token的直接识别，避免错误拆分
   - 完善对话模板格式化，支持多角色对话和Qwen3标准格式
   - 根据思考模式动态设置采样参数（temperature、top_p、top_k）
   - 实现思考内容解析功能，提取<think>...</think>标记内容
   - 优化EOS检测，确保正确识别<|im_end|>标记并终止生成
   - 添加详细日志记录，便于排查分词和生成问题
   - 实现防循环机制，避免生成重复内容

2. **模型输出张量处理**
   - 支持多维度输出张量的动态处理，适应不同模型结构
   - 自动检测输出张量的维度和形状，支持二维和三维输出格式
   - 在日志中记录输出张量的形状信息，便于调试和分析
   - 实现强健的异常处理机制，对不支持的输出形状提供清晰的错误信息

3. **ONNX Runtime内存优化**
   - 启用内存模式优化（Memory Pattern Optimization），减少内存碎片化
   - 保留完整上下文信息，不截断输入token，确保RAG应用效果最佳
   - 实现精确的token数量内存影响评估，并在日志中详细记录
   - 自动识别模型量化精度（从模型名称和配置文件），并设置相应的推理模式
   - 对于int8量化模型，自动启用int8推理和禁用内存竞技场，减少内存消耗
   - 动态调整线程数为系统CPU核心数的一半，平衡性能和内存使用
   - 在生成过程中增加内存监控频率，每5次迭代检查一次
   - 当系统可用内存低于500MB时自动终止生成过程，避免OOM崩溃
   - 开启ONNX Runtime详细日志，便于分析内存使用情况
   - 对于内存需求超过6GB的长文本处理，显示警告信息

4. **Token数量与内存消耗分析**
   - 量化模型（int8）在推理过程中会将数据转换为float32格式进行计算
   - 每个token的KV缓存内存占用计算公式：隐藏层大小 * 2(K和V) * 4字节(float32) * 层数
   - 对于标准Transformer模型（隐藏层大小1024，24层），每个token的KV缓存约占用196KB
   - 20,000个token的KV缓存需要约3.92GB内存，再加上模型参数和中间计算结果，总内存需求可能超过5GB
   - 对于移动设备，处理长文本时内存消耗是主要瓶颈，即使模型本身只有600MB

2. **处理速度**
   - **GPU加速支持**：
     * EmbeddingModel和LocalLLM均支持GPU加速
     * 支持NNAPI、OpenCL、CUDA等多种GPU后端
     * GPU加速失败时自动降级到CPU模式，确保稳定性
     * 用户可在设置中开启/关闭GPU加速，实时生效
   - 优化分块策略，减少不必要的处理
   - 使用TokenizerManager统一分词策略，提高检索准确性
   - 缓存常用数据，减少重复计算

3. **用户体验**
   - 实现进度显示，提供详细的处理状态
   - 支持取消操作，避免长时间等待
   - 添加防锁屏功能，处理大文件时保持屏幕常亮
   - 优化UI响应速度，使用Handler更新UI

### 3.3 界面与交互优化

1. **界面设计**
   - 使用Spinner下拉框替代AutoCompleteTextView，提升一致性
   - 支持文本缩放功能，方便用户阅读
   - 提供Markdown渲染，美化AI回答显示
       - 优化流式响应的Markdown渲染逻辑，采用"流式纯文本+最终完整渲染"策略
       - 在数据流式接收过程中使用纯文本显示，避免频繁渲染导致的格式破坏
       - 在数据接收完成后（onSuccess回调中）进行一次完整的Markdown渲染
       - 确保代码块完整性，通过检测和修复不完整的代码块标记
       - 保持滚动位置一致性，确保用户阅读体验不受内容更新影响
       - 确保链接可点击，通过设置LinkMovementMethod实现
       - 添加渲染性能监控，记录渲染时间和内容长度
       - 优化代码块样式，区分代码块和行内代码的样式
         - 代码块（多行代码）：使用深色背景（#282c34）和浅色文本（#abb2bf），类似VSCode的One Dark主题
         - 行内代码（文本中的代码片段）：不使用背景色，仅使用等宽字体区分，保持文本流畅性
         - 增加代码块内边距和块间距，提高可读性
         - 使用蓝紫色（#5c6bc0）美化引用块，增强视觉层次
   - 美化滚动条箭头显示，使用scaleType="fitCenter"和padding属性

2. **交互优化**
   - 支持"转笔记"功能，方便将AI回答保存为知识库笔记
   - 自定义文本选择菜单，确保"转笔记"选项只出现一次，避免重复显示
   - 知识库已存在时提供"追加"、"覆盖"和"取消"三个选项
   - 模型选择对话框支持"记住选择"功能
   - 支持长按文本选择和上下文菜单操作（复制、全选、转笔记等）
   - 提供调试模式开关，控制调试信息的显示

3. **错误处理**
   - 提供详细的错误提示和日志记录
   - 支持模型加载失败时自动降级到CPU模式
   - 处理XML解析崩溃问题，优化MIME类型检测
   - 使用适当的替代方法处理API兼容性问题

### 3.4 进度消息优化

- **避免重复显示相同状态信息**：在进度显示中避免重复显示相同的状态信息，确保用户界面清晰易读。
- **使用统一格式显示处理进度**：
  - 顶部标签显示 文本提取阶段：使用格式`"正在提取文本 (N/M): xxx"`
  - 顶部标签显示 向量化阶段：使用格式`"正在生成向量 (N/M): xxx%"`
  - 文本框显示日志区域显示：向量化进度显示：使用格式`"向量化进度..........10%..........20%..."`，每个点代表一个处理完成的文本块
  - 这种统一的格式提供了清晰的进度信息，并确保进度解析逻辑正确工作。
- **保留历史进度信息**：
  - 知识库构建完成后不清空进度框，保留之前的日志信息
  - 实现自动滚动到最新进度信息，同时允许用户上滚查看历史记录
  - 在任务完成时追加最终状态，而不是替换整个进度内容

### 3.5 安全性考虑

1. **数据安全**
   - 所有数据存储在本地，不依赖云服务
   - API密钥存储在本地配置中，不会上传到服务器
   - 支持多种API服务器，用户可自由选择

2. **权限管理**
   - 只请求必要的存储权限
   - 提供权限说明对话框，解释权限用途
   - 支持运行时权限请求，符合现代Android开发标准

### 3.6 维护与扩展性

1. **代码组织**
   - 遵循模块化设计，各组件职责明确
   - 使用统一的日志标签格式，便于问题定位
   - 关键方法添加详细注释，说明功能和参数
   - 使用同步机制（synchronized块）确保线程安全

2. **扩展性**
   - 支持多种文档格式，可扩展支持更多格式
   - 支持多种嵌入模型，可根据需求替换
   - 提供配置接口，可根据需求调整参数

3. **版本管理**
   - 使用构建时间作为版本号，格式为yyyyMMddHHmmss
   - 在关于对话框中显示版本信息
   - 使用日志记录关键操作，便于问题排查

### 3.7 嵌入模型管理优化

为了优化应用体积和确保用户体验，我们对嵌入模型管理进行了以下优化：

1. **模型路径处理优化**：
   - 简化了`EmbeddingModelHandler`中的`loadTokenizerAndConfig`方法，使用更直接的路径处理逻辑
   - 优化了tokenizer.json文件的定位策略，使用与模型目录结构一致的方式
   - 当使用量化模型（如bge-m3_static_quant_INT8）时，自动在同级目录中查找原始模型（如bge-m3）的tokenizer.json
   - 增强了日志记录，便于跟踪模型和分词器的加载过程

2. **移除内置嵌入模型**：
   - 移除了应用中内置的嵌入模型（如xim-roberta模型），减少应用体积
   - 修改EmbeddingModelUtils类，移除基于模型名称推断维度的逻辑，仅保留通过加载模型确定维度的方法
   - 确保应用只使用用户自定义的模型，提高灵活性和可扩展性

2. **模型选择逻辑优化**：
   - 修改BuildKnowledgeBaseFragment中的createKnowledgeBase方法，确保选择的嵌入模型存在
   - 添加错误处理机制，当选择的模型不存在时向用户显示提示信息
   - 增强模型选择对话框，确保显示的模型与数据库预期维度匹配

3. **数据库操作优化**：
   - 在SQLiteVectorDatabaseHandler中添加clearDatabase方法，支持覆盖现有知识库的功能
   - 在KnowledgeBaseBuilderService的startBuildKnowledgeBase方法中添加overwrite参数，允许选择覆盖现有知识库
   - 在TextChunkProcessor中更新processFilesAndBuildKnowledgeBase方法，支持覆盖选项

4. **错误处理增强**：
   - 添加模型不存在时的错误提示和处理逻辑
   - 实现模型加载失败时的优雅降级和用户反馈
   - 确保在模型选择过程中提供清晰的用户指导

这些优化措施显著提高了应用的稳定性和用户体验：
- 减小了应用体积，降低了安装和更新的网络消耗
- 提高了应用的灵活性，允许用户使用自己选择的嵌入模型
- 增强了错误处理能力，提供更友好的用户反馈
- 改进了知识库管理功能，支持覆盖现有知识库的选项

### 3.8 模型使用跟踪

为了防止模型过早卸载，我们实现了模型使用跟踪机制：

1. **使用状态跟踪**：
   - 在EmbeddingModelManager中添加markModelInUse()和markModelNotInUse()方法
   - 在BuildKnowledgeBaseFragment、RagQaFragment和KnowledgeNoteFragment中集成这些方法
   - 确保模型在需要时保持加载状态，不会被过早卸载

2. **模型路径更新**：
   - 在DatabaseMetadata类中添加setEmbeddingModel()方法，支持模型路径更新
   - 确保知识库元数据中的模型路径始终是最新的

这些优化确保了模型在使用过程中不会被意外卸载，提高了应用的稳定性和用户体验。

### 3.9 向量维度匹配与性能优化

为了解决向量维度不匹配问题并提升模型处理性能，我们实现了以下优化：

1. **向量维度匹配优化**：
   - 修改`EmbeddingModelHandler.getEmbeddingDimension()`方法，优先从配置文件获取维度信息
   - 添加带维度参数的`SQLiteVectorDatabaseHandler`构造函数，确保元数据中保存正确的维度信息
   - 修改`TextChunkProcessor.processFilesAndBuildKnowledgeBase`方法，在构建知识库时明确设置向量维度
   - 将默认向量维度从768改为1024，适应bge-m3等大型模型的需求

2. **CPU并发性能优化**：
   - 修改`EmbeddingModelHandler.loadOnnxModel`方法，动态设置线程数为系统CPU核心数的一半
   - 替换原有的固定单线程设置（`setIntraOpNumThreads(1)`和`setInterOpNumThreads(1)`）
   - 添加详细日志记录，显示可用CPU核心数和实际使用的线程数
   - 确保至少使用1个线程，保证在低端设备上也能正常工作

这些优化带来的主要好处：
- 解决了构建知识库和查询时可能出现的"向量维度不匹配"错误
- 显著提升了向量化处理性能，特别是在多核设备上
- 增强了系统对不同模型的适应性，特别是对bge-m3等大型模型的支持
- 保持了与现有功能的兼容性，不会破坏已创建的知识库

### 3.10 通知栏和电池优化管理

为了提升用户体验和应用在后台运行时的稳定性，我们对通知栏和电池优化管理进行了以下优化：

1. **通知栏管理优化**：
   - 修改`KnowledgeBaseBuilderService`的`onStartCommand`方法，不再在服务启动时就立即显示通知栏
   - 在`startBuildKnowledgeBase`方法中才启动前台服务并显示通知栏，确保只有在实际开始构建知识库时才显示通知
   - 修改`stopSelfDelayed`方法，确保服务停止时立即移除通知栏，并将延迟时间从5秒减少到1秒
   - 添加更详细的通知内容，显示当前处理阶段和进度信息

2. **电池优化管理**：
   - 在`KnowledgeBaseBuilderService`中添加`onBuildCompleted`回调，用于通知知识库构建完成或取消
   - 在`BuildKnowledgeBaseFragment`中实现回调处理，确保在知识库构建完成或取消时恢复电池优化设置
   - 添加`enableKeepScreenOn`方法，用于控制屏幕常亮状态
   - 确保在知识库构建完成、取消或失败时都会恢复电池优化设置和屏幕常亮状态

3. **进度回调优化**：
   - 添加`onTextExtractionComplete`和`onVectorizationComplete`回调，用于通知文本提取和向量化完成
   - 在回调中传递处理的文本块数量和向量数量，提供更详细的进度信息
   - 在UI中显示更详细的处理状态，包括文本提取和向量化的完成情况

这些优化带来的主要好处：
- 改善了用户体验，通知栏只在实际需要时显示，避免不必要的干扰
- 提高了应用在后台运行时的稳定性，通过合理管理电池优化和屏幕常亮状态
- 提供了更详细的进度信息，让用户清楚了解知识库构建的各个阶段
- 确保了资源的及时释放，避免不必要的电池消耗和系统资源占用

### 3.11 流式响应处理优化

为了提升用户体验，特别是在接收LLM流式响应时，我们对RagQaFragment中的响应处理逻辑进行了以下优化：

1. **防止屏幕闪烁**：
   - 使用累积响应的方式，通过StringBuilder收集所有流式响应片段
   - 实现批量更新机制，只有当内容变化足够大或时间间隔足够长时才更新UI
   - 使用阈值控制（MIN_CHAR_CHANGE和UPDATE_INTERVAL），减少UI更新频率
   - 使用Markwon的高级API（toMarkdown和setParsedMarkdown）进行预处理和渲染，避免渲染过程中的闪烁

2. **取消自动滚动**：
   - 移除了自动滚动功能，避免内容更新时的跳动问题
   - 允许用户自行控制滚动位置，提供更好的阅读体验
   - 避免了在阅读时被强制拖拽到底部的困扰

3. **防止重复输出**：
   - 重构onSuccess回调，避免在流式响应完成后再次显示完整响应
   - 使用标记变量（modelTitleAdded）确保模型回答标题只添加一次
   - 优化lastDisplayedResponse记录机制，确保内容不会重复显示

4. **Markdown渲染优化**：
   - 增强Markwon初始化配置，添加AbstractMarkwonPlugin插件支持
   - 使用LinkMovementMethod确保链接可点击并正确处理文本滚动
   - 使用Markwon的toMarkdown和setParsedMarkdown方法替代简单的setMarkdown，提高渲染效率
   - 优化模型回答标题的显示格式，使用更清晰的Markdown语法
   - 确保代码块、列表等Markdown格式正确显示，提升可读性

这些优化带来的主要好处：
- 显著减少了屏幕闪烁，提供更平滑的阅读体验
- 用户可以自由控制滚动位置，不再被强制拖拽到底部
- Markdown格式渲染更加精确和美观，提升了内容的可读性
- 确保文本视图始终滚动到最新内容，同时避免频繁滚动带来的视觉干扰
- 消除了流式响应完成后的重复输出问题
- 提高了长文本响应的渲染性能和用户体验
- 改善了Markdown内容的显示效果，使模型回答更易于阅读

### 3.12 分块参数范围优化

为了提高应用在处理不同类型文本时的灵活性，我们对分块参数进行了以下优化：

1. **最小分块限制范围扩展**：
   - 将最小分块限制的上限从 100 字符扩展到 200 字符
   - 保持下限为 10 字符，确保分块不会过小
   - 修改了 `SettingsFragment` 中的验证逻辑和提示信息

2. **优化目的**：
   - 增强对特定类型文本的处理能力，如学术论文、技术文档等包含长句子的文本
   - 允许用户保留更长的语义完整段落，提高检索质量
   - 适应不同语言的语句结构特点，如中文长句和英文长句

3. **实现方式**：
   - 修改 `SettingsFragment.java` 中的验证逻辑
   - 更新用户界面提示信息，清晰地显示新的参数范围

这些优化使应用能够更好地适应不同类型的文本内容，特别是对于包含长句子或复杂表达的文档，可以保持更好的语义完整性。

### 3.13 本地LLM推理功能

为了支持在设备上进行本地大语言模型推理，我们实现了以下功能和组件：

1. **本地模型处理器架构**：
   - 实现`LocalLLMHandler`类作为主处理器，负责模型加载、卸载和类型判断
   - 新增`LocalLLMOnnxHandler`类专门处理ONNX模型的推理逻辑
   - 采用策略模式设计，根据模型类型动态选择合适的处理器
   - 支持未来扩展其他类型模型（如PyTorch等）的处理器
   - 实现内存管理机制，避免重复加载模型
   - 支持模型自动卸载，释放内存资源
   - 自动提供`attention_mask`输入，支持Transformer类模型
   - 智能识别模型输入节点名称，适配不同模型结构
   - 自动检测模型量化类型和外部数据格式，优化推理性能
   - 根据模型特性自动调整内存使用和线程配置

2. **推理引擎架构**：
   - 实现`InferenceEngine`接口，支持多种推理引擎实现
   - `GenAIInferenceEngine`：基于ONNX Runtime GenAI的高级推理引擎
     * 使用ONNX Runtime GenAI的高级API，简化模型推理流程
     * 自动管理模型生命周期和内存资源
     * 支持流式文本生成和实时响应
     * 提供更好的性能和稳定性
   - `OnnxRuntimeInferenceEngine`：基于传统ONNX Runtime的推理引擎
     * 保持与现有代码的兼容性
     * 支持低级别的模型控制和优化
   - 根据配置动态选择推理引擎，支持运行时切换
   - 提供统一的推理接口，确保不同引擎间的无缝切换

3. **推理引擎配置管理**：
   - 在`ConfigManager`中添加`KEY_USE_ONNX_GENAI`配置键
   - 默认启用ONNX Runtime GenAI引擎（`true`）
   - 在设置界面提供推理引擎切换开关
   - 支持实时更新推理引擎配置，无需重启应用
   - 通过`LocalLlmHandler.updateEngineFromConfig()`方法动态切换引擎

4. **本地模型适配器**：
   - 实现`LocalLlmAdapter`类，符合现有API适配器接口
   - 提供与在线模型一致的调用方式，确保无缝集成
   - 支持流式响应，提供实时反馈
   - 实现单例模式，确保模型资源的高效利用

5. **模型工厂扩展**：
   - 在`LlmModelFactory`中添加`LOCAL`类型支持
   - 扩展API类型检测逻辑，支持本地模型识别
   - 实现本地模型调用逻辑，包括同步和异步调用
   - 与现有模型工厂方法无缝集成

6. **GPU加速支持**：
   - **EmbeddingModel GPU加速**：
     * 在`EmbeddingModelHandler`中实现GPU加速支持
     * 支持NNAPI、OpenCL、CUDA等多种GPU加速方式
     * GPU加速失败时自动降级到CPU模式
     * 通过`EmbeddingModelManager`统一管理GPU设置
   - **LocalLLM GPU加速**：
     * 在`LocalLlmHandler`中实现GPU加速支持
     * 支持多种GPU加速后端的自动检测和启用
     * 通过`LocalLlmAdapter`提供GPU设置更新接口
   - **统一GPU配置管理**：
     * 在设置界面提供GPU加速开关
     * 实时更新GPU设置，无需重启应用
     * 在`MainActivity`中监听设置变更并同步到各组件
   - **错误处理和日志**：
     * 提供详细的GPU加速日志记录
     * 使用异常处理确保GPU加速失败不会导致应用崩溃
     * 通过`GPUErrorHandler`统一处理GPU相关错误
   - **GPU故障诊断与排除**：
     * `GPUDiagnosticTool`：提供详细的GPU硬件信息、系统兼容性检查和HarmonyOS特殊处理
     * `GPUConfigChecker`：验证AndroidManifest.xml配置、权限设置和硬件特性支持
     * 应用启动时自动执行GPU配置检查，提前发现配置问题
     * GPU加速失败时自动生成详细诊断报告，包含系统信息、硬件特性、配置状态
     * 针对华为HarmonyOS设备提供专门的优化建议和配置指导
     * 创建`GPU_TROUBLESHOOTING_GUIDE.md`故障排除指南，包含常见问题FAQ和修复方案
   - **HarmonyOS GPU适配优化**：
     * `HarmonyOSGPUAdapter`：专门处理HarmonyOS设备的GPU加速优化
     * 实现HarmonyOS特定的OpenCL缓存配置和Mali GPU优化
     * 支持Vulkan API的设备本地内存配置和FP16精度优化
     * 检测HarmonyOS GPU访问权限，包括ohos.permission.USE_GPU权限
     * 提供系统级GPU调度策略和内存管理优化
     * 支持Vulkan计算支持检测和配置参数调整
   - **Mali GPU专项优化**：
     * `MaliGPUOptimizer`：针对Mali-G610等Mali GPU架构的性能优化
     * 实现Mali GPU架构检测，支持从系统路径和OpenGL渲染器获取GPU信息
     * 提供Mali GPU特定的ONNX Runtime配置，包括线程数、内存带宽、卷积优化
     * 支持Mali GPU性能监控，读取GPU使用率、频率和温度
     * 实现GPU频率调节器优化，设置性能模式和最小频率
     * 提供Mali GPU内存分配策略，包括内存池大小、压缩和垃圾回收优化
     * 检测Mali GPU特性支持，包括OpenCL、Vulkan、计算着色器和FP16支持
   - **OpenGL ES计算加速**：
     * `OpenGLESComputeAccelerator`：使用OpenGL ES 3.2计算着色器实现GPU加速
     * 支持OpenGL ES计算着色器的初始化和支持检测
     * 提供基础计算和矩阵乘法的计算着色器程序
     * 实现GPU内存缓冲区管理和计算结果读取
     * 支持计算着色器资源的清理和释放

5. **UI交互优化**：
   - 在API URL选择器中添加固定的“本地”选项
   - 修改`ApiUrlAdapter`确保“本地”选项不能被删除
   - 优化模型选择逻辑，自动加载本地可用模型列表
   - 在选择“本地”选项时不需要API Key

5. **ONNX模型推理实现**：
   - 实现完整的自回归生成算法，支持逐token生成
   - 支持温度采样（Temperature Sampling）调节生成的随机性
   - 实现Top-K采样，控制每步生成时考虑的候选token数量
   - 实现Top-P（核心采样），确保生成文本的多样性和质量
   - 支持思考模式（Thinking Mode）的处理，可提取思考内容和最终回答
   - 实现错误处理和回退机制，在自回归生成失败时尝试单步推理
   - 自动适配不同维度的模型输出，支持二维和三维logits输出格式
   - 提供详细的生成日志，包括token生成速度和总耗时统计
   - 使用try-finally块确保所有ONNX资源被正确释放，防止JNI全局引用表溢出
   - 实现克隆输出数组，避免直接引用JNI对象导致的内存问题
   - 在资源释放后将引用设为null，帮助垃圾回收器更有效地回收内存

6. **JNI资源管理与内存优化**：
   - 实现`close()`方法，确保线程池和相关资源被正确关闭
   - 使用try-finally块确保所有OnnxTensor和OrtSession.Result对象都被及时释放
   - 在每次推理迭代后立即释放资源，避免JNI全局引用表溢出
   - 在资源释放后将引用设置为null，帮助垃圾回收器更有效地回收内存
   - 使用深度克隆数组（System.arraycopy）而非简单的.clone()，完全断开与JNI对象的联系
   - 定期触发垃圾回收，帮助清理JNI引用（每生成一定数量的token后触发）
   - 使用类级别的Handler对象，减少重复创建new Handler(Looper.getMainLooper())的次数
   - 实现内存使用情况的周期性打印，方便调试和监控内存泄漏
   - 优化对象创建，尽量复用已有对象，减少内存分配频率
   - 使用modelInputs.clear()清除之前的输入，而不是每次都创建新的Map
   - 限制最大token数，避免过多循环导致内存问题
   - 实现异常安全的资源释放，即使在出错时也能正确关闭资源
   - 实现单步推理回退机制，在自回归生成失败时仍能确保资源正确释放
   - 实现JNI引用计数器，跟踪全局引用数量
   - 设置JNI引用阈值（MAX_JNI_REFERENCES），当引用数量超过阈值时强制清理
   - 周期性清理JNI引用，防止引用数量持续增长（每生成JNI_CLEANUP_INTERVAL个token清理一次）
   - 使用弱引用缓存（WeakReference）存储对象，减少重复创建同样对象的次数
   - 实现clearWeakCache方法，定期清理已被垃圾回收的弱引用
   - 优化异常处理，修复重复异常变量导致的编译错误
   - 实现批处理逻辑（BATCH_SIZE参数控制），减少JNI调用频率
   - 在关键点记录JNI引用计数，便于监控和调试引用泄漏问题

7. **Token解码与文本生成优化**：
   - 实现专用的`decodeToken`方法，正确处理Qwen模型的特殊token编码
   - 使用映射表处理多种特殊token，包括空格前缀、换行符和Unicode字符
   - 增强对中文和其他Unicode字符的支持，通过Unicode块识别不同字符类型
   - 添加详细的日志记录，跟踪token解码过程，便于调试乱码问题
   - 优化分词实现，添加对tokenizer初始化状态的检查
   - 实现UTF-8编码的字节序列处理，支持特殊字符的正确解码
   - 添加对十六进制编码字节的处理，支持`<0x..>`格式的特殊token
   - 增强错误处理，对未找到的token映射提供明确的警告日志
   - 实现空格前缀（如`Ġ`）的特殊处理，确保文本格式正确
   - 支持罗马数字和其他特殊符号的映射，提高文本可读性
   - 优化标点符号处理，支持多种标点的正确显示
   - 在token生成过程中添加详细日志，记录token ID和解码前后的内容
   - 实现对批量token的高效处理，减少重复解码开销

7. **模型目录结构**：
   - 使用用户可配置的模型路径
   - 每个模型使用独立目录，目录名即为模型名
   - 支持动态发现新添加的模型，无需重启应用

这些功能带来的主要好处：
- 实现了完全离线的LLM推理能力，无需依赖云服务
- 保持了与在线模型一致的用户体验，无需学习新的交互方式
- 提高了数据隐私保护能力，敏感数据只在本地处理
- 降低了使用成本，无需支付API调用费用
- 提供了灵活的模型选择机制，支持用户自定义模型
- 代码结构清晰，便于维护和扩展，支持未来添加新的模型类型
- 实现了高质量的文本生成算法，生成效果接近在线 API 服务
- 解决了JNI全局引用表溢出问题，显著提高了应用稳定性
- 优化了内存使用效率，减少内存泄漏和资源浪费
- 提高了长时间运行时的稳定性，避免由于资源耗尽导致的崩溃

### 3.12 Qwen3 模型推理优化

为了提升 Qwen3 模型的推理质量和性能，我们对 ONNX 模型推理逻辑进行了全面优化：

1. **Token 采样策略**：
   - 实现温度采样（Temperature Sampling）机制，通过可调节的温度参数控制文本生成的随机性
   - 实现 Top-K 采样策略，只保留概率最高的 K 个候选 token，大幅提高生成文本的质量
   - 根据是否开启思考模式动态调整采样参数，思考模式下使用 temperature=0.6, topK=20，非思考模式下使用 temperature=0.7, topK=5
   - 使用累积概率采样方法，确保采样过程符合概率分布

2. **分词优化**：
   - 重构 Qwen3 模型的分词逻辑，解决内存不足问题
   - 利用 TokenizerManager 中的 BertTokenizer 进行分词，避免重复实现
   - 实现分词备选方案，当 BertTokenizer 不可用时回退到原始分词方法
   - 优化特殊 token 处理，正确识别和处理 Qwen3 模型的特殊 token
   - 实现文本分段处理，提高分词效率和准确性
   - 添加详细日志记录，便于调试和比较分词结果
   - 统一分词策略，确保 RAG 应用和 LLM 推理使用一致的分词逻辑

### 3.13 分词器路径管理优化

为了解决分词器初始化路径错误导致的问题，我们对分词器路径管理进行了全面优化：

1. **路径一致性优化**：
   - 修正 `LocalLLMOnnxHandler` 中的分词器初始化路径，使用用户选择的模型实际目录
   - 使用 `ConfigManager.getModelPath(context)` 获取模型根目录路径，而不是硬编码的应用内部存储目录
   - 根据模型类型和会话信息构建完整的模型目录路径
   - 添加异常处理，提高代码健壮性

2. **分词器文件检测与加载**：
   - 改进分词器文件检测逻辑，正确处理 `tokenizer.json`、`tokenizer_config.json` 和 `special_tokens_map.json` 等文件
   - 实现智能路径推断，当会话信息中包含模型名称时自动提取
   - 当无法从会话中提取模型名称时，使用模型类型作为备选
   - 添加详细日志，记录分词器初始化过程和路径信息

3. **分词器实例管理**：
   - 使用 `TokenizerManager` 的单例模式管理分词器实例，避免重复初始化
   - 实现分词器状态检查，在使用前验证分词器是否正确初始化
   - 实现分词器重置机制，支持在模型切换时重新初始化
   - 优化内存使用，避免重复加载分词器文件

这些优化措施显著提高了应用的稳定性和用户体验：
- 解决了分词器初始化失败导致的推理异常问题
- 提高了代码的健壮性，增强了错误处理能力
- 改进了路径管理逻辑，支持灵活的模型目录结构
- 增强了日志记录，便于问题排查和调试

### 3.14 Rust 分词器集成优化

为了提高分词性能和跨平台兼容性，我们集成了基于 Rust 的 Hugging Face Tokenizers 库：

1. **JNI 桥接层设计**：
   - 实现 `RustTokenizerJNI` 类，通过 JNI 调用 Rust 实现的分词功能
   - 使用 `System.loadLibrary` 动态加载 `tokenizers_ffi` 和 `tokenizers_jni` 原生库
   - 实现异常处理机制，确保 JNI 调用失败时能够优雅降级到 Java 实现
   - 添加资源释放方法，防止内存泄漏

2. **多架构支持**：
   - 为 ARM 架构（arm64-v8a 和 armeabi-v7a）提供完整的 Rust 实现
   - 为 x86 和 x86_64 架构提供存根库，确保编译成功
   - 在运行时检测架构类型，自动回退到 Java 实现（针对不支持的架构）
   - 使用 CMake 配置文件动态选择正确的库路径

3. **接口适配设计**：
   - 实现 `RustBertTokenizer` 类，继承自 `BertTokenizer` 接口
   - 使用适配器模式将 Rust 分词器功能转换为应用所需的接口
   - 保持与现有 Java 实现的 API 兼容性，确保无缝集成
   - 实现正确的资源管理，确保分词器实例在不再使用时被释放

4. **编译与打包优化**：
   - 使用 Gradle 任务自动编译 Rust 代码并生成 `.so` 文件
   - 配置 CMake 构建脚本，处理不同架构的编译需求
   - 实现存根库，解决不兼容架构的链接问题
   - 设置 JAR 文件的重复处理策略，避免打包错误

这些优化措施带来的主要好处：
- 显著提高了分词性能，特别是在处理大型文本时
- 增强了跨平台兼容性，支持多种 Android 设备架构
- 保持了与现有代码的兼容性，无需修改上层应用逻辑
- 提供了优雅的降级机制，确保在不支持的环境中仍能正常工作
- 实现了更高效的内存管理，减少了资源消耗

### 3.15 分词器架构重构

为了简化分词器架构并提高代码的可维护性，我们对分词器相关组件进行了全面重构：

1. **架构简化**：
   - 移除了不必要的 `BertTokenizer` 和 `Tokenizer` 接口，减少了架构复杂性
   - 将 `RustBertTokenizer` 重命名为 `HuggingfaceTokenizer`，更准确地反映其功能
   - 合并 `HuggingfaceTokenizerJNI` 和 `HuggingfaceTokenizer` 类，将 JNI 方法直接集成到分词器类中
   - 实现 `AutoCloseable` 接口，确保资源正确释放

2. **TokenizerManager 增强**：
   - 将 `TokenizerManager` 作为应用中分词器的唯一入口点
   - 内部直接使用 `HuggingfaceTokenizer` 进行分词操作
   - 维护单例模式，确保全局只有一个分词器实例
   - 提供统一的分词方法和词汇表管理功能

3. **多架构支持**：
   - 改进 CMakeLists.txt 配置，增加对 x86 和 x86_64 架构的完全支持
   - 为所有支持的架构配置正确的库路径：
     - arm64-v8a: aarch64-linux-android
     - armeabi-v7a: armv7-linux-androideabi
     - x86: i686-linux-android
     - x86_64: x86_64-linux-android
   - 移除存根库实现，使用完整的 Rust 库支持所有架构

4. **JNI 实现优化**：
   - 更新 tokenizers_jni.cpp 中的 JNI 函数名，与新的 `HuggingfaceTokenizer` 类匹配
   - 更新日志标签从 "RustTokenizer" 到 "HuggingfaceTokenizer"
   - 保持 JNI 函数签名不变，确保与原生库的兼容性

5. **EmbeddingModelHandler 集成**：
   - 更新 `EmbeddingModelHandler` 类，将 tokenizer 字段类型从 `Tokenizer` 改为 `TokenizerManager`
   - 修改 tokenizeText 方法，确保使用 TokenizerManager 进行分词
   - 简化一致性分词策略的处理逻辑

这次重构带来的主要改进：

- **架构简化**：移除了不必要的接口层，使代码结构更清晰
- **命名一致性**：使用 "Huggingface" 前缀替代 "Rust"，更准确地反映底层实现
- **多架构支持**：增强了对 x86 和 x86_64 架构的支持，提高了应用的兼容性
- **代码简化**：合并了 JNI 和分词器实现，减少了类的数量
- **统一入口点**：确保所有分词操作都通过 TokenizerManager 进行，提高了一致性

这些改进显著提高了代码的可维护性和可扩展性，同时保持了与现有功能的兼容性。通过增强对多架构的支持，应用能够在更多的设备上运行，提高了用户体验。

### 3.16 TokenizerManager 集成优化

为了确保分词器正确利用用户选择的模型，并在模型切换时正确管理资源，我们对 TokenizerManager 和 EmbeddingModelHandler 的集成进行了全面优化：

1. **配置文件解析增强**：
   - 实现从 config.json 文件中读取分词器类型的功能
   - 支持自动识别并加载不同模型类型的分词器（BERT、RoBERTa、Qwen3 等）
   - 建立分词器类型与模型类型的映射关系，确保正确匹配

2. **错误处理增强**：
   - 在分词器加载失败时抛出异常，而不是默默回退到简单分词器
   - 在 TokenizerManager 中添加详细的错误日志，包含分词器类型、模型路径和失败原因
   - 实现分词器初始化状态检查，确保在使用前已正确初始化

3. **资源管理优化**：
   - 在 EmbeddingModelHandler 中添加 reset 方法，确保在切换模型时释放分词器资源
   - 在 loadTokenizerAndConfig 方法中添加资源清理逻辑，防止内存泄漏
   - 实现安全的会话恢复机制，在 ONNX 会话失效时自动尝试恢复

4. **一致性分词策略**：
   - 在 TokenizerManager 中添加 setUseConsistentTokenization 方法，支持全局设置一致性分词策略
   - 在 EmbeddingModelHandler 中同步设置分词器的一致性分词策略
   - 确保在知识库构建和查询过程中使用相同的分词逻辑

5. **模型名称提取增强**：
   - 实现从模型路径、config.json 和 tokenizer.json 文件中提取模型名称的功能
   - 添加对 BGE-M3 等特定模型的识别支持
   - 根据 hidden_size 推断模型大小（Tiny、Mini、Small、Medium、Large）

6. **调试信息增强**：
   - 添加详细的向量生成调试信息，包含向量维度、处理时间和向量样本
   - 实现 getVectorDebugInfo 方法，提供格式化的调试信息
   - 在日志中记录分词结果和向量范数信息，便于排查问题

7. **会话管理优化**：
   - 实现 ONNX 会话状态管理，包括未初始化、加载中、就绪和错误状态
   - 添加会话重试机制，在会话失效时自动尝试恢复
   - 使用同步锁确保会话操作的线程安全

这些优化提高了应用的稳定性和可靠性，确保分词器能够正确地与用户选择的模型集成，并在模型切换时正确管理资源。通过增强的错误处理和调试信息，开发者可以更容易地识别和解决分词器相关的问题。

### 3.17 本地LLM推理中断功能

为了提升用户体验，特别是在处理长时间推理任务时，我们实现了本地LLM推理的中断功能：

1. **停止标志机制**：
   - 在`LocalLlmHandler`类中添加`AtomicBoolean shouldStopInference`原子布尔变量
   - 实现`stopInference()`方法设置停止标志
   - 实现`shouldStopInference()`方法检查停止状态
   - 实现`resetStopFlag()`方法重置停止标志

2. **推理循环中断检查**：
   - 在`LocalLLMOnnxHandler`的`inferenceStream`方法中添加`LocalLlmHandler`参数
   - 在自回归生成循环的开始处添加停止检查逻辑
   - 当检测到停止标志时，记录日志、调用错误回调并返回
   - 确保推理过程能够及时响应停止请求

3. **适配器层集成**：
   - 在`LocalLlmAdapter`的`stopGeneration`方法中调用`localLlmHandler.stopInference()`
   - 确保停止功能与现有API适配器接口保持一致
   - 提供统一的停止接口，便于上层调用

4. **UI层停止逻辑**：
   - 在`RagQaFragment`的`handleSendStopClick`方法中添加本地模型停止逻辑
   - 根据当前API URL判断是否为本地模型（"local"）
   - 调用`LocalLlmAdapter.getInstance(requireContext()).stopGeneration()`停止本地LLM推理
   - 添加异常处理和日志记录，确保停止操作的稳定性

5. **线程安全设计**：
   - 使用`AtomicBoolean`确保停止标志的线程安全访问
   - 在推理开始时重置停止标志，避免上次操作的影响
   - 在推理过程中定期检查停止标志，确保及时响应

这些功能带来的主要好处：
- 用户可以随时中断长时间运行的本地LLM推理任务
- 避免了因无法停止推理而导致的应用无响应问题
- 提供了与在线模型一致的用户体验
- 增强了应用的可控性和用户友好性
- 通过线程安全的设计确保了功能的稳定性

### 3.18 RAG界面模型选择记忆功能

为了提升用户体验，避免每次使用时都需要重新选择模型，我们实现了模型选择的记忆功能：

1. **模型选择保存机制**：
   - 在`RagQaFragment`的`fetchModelsForApi`方法中实现模型选择的保存和恢复
   - 在刷新模型列表时保存当前选择的模型名称
   - 在新的模型列表加载完成后，自动恢复之前选择的模型

2. **配置持久化**：
   - 使用`ConfigManager`保存用户最后选择的模型名称
   - 模型选择信息存储在`.config`文件中，确保应用重启后仍能恢复
   - 支持不同API地址对应不同的模型选择记忆

3. **智能匹配逻辑**：
   - 在模型列表更新时，查找与保存的模型名称匹配的选项
   - 如果找到匹配的模型，自动设置为选中状态
   - 如果未找到匹配项，保持默认选择（通常是第一个模型）

4. **用户交互优化**：
   - 为模型Spinner添加选择监听器，实时保存用户的选择
   - 在用户切换模型时立即保存新的选择
   - 提供详细的日志记录，便于调试和用户反馈

这些功能显著改善了用户体验：
- 避免了每次使用时都需要重新选择模型的繁琐操作
- 提高了应用的易用性和用户满意度
- 确保了用户偏好设置的持久化保存
- 支持多API环境下的独立模型选择记忆

### 3.19 RAG 界面布局优化与思考模式功能整合

为了提升用户体验和操作便利性，我们对RAG问答界面进行了布局优化，并将思考模式功能从设置页面迁移到RAG界面。

#### 3.19.1 界面布局优化

**布局调整目标**：
- 调整大模型下拉框宽度，使其与知识库选择框对齐，提升视觉一致性
- 在界面右侧空白区域添加思考模式复选框，充分利用界面空间
- 优化控件间距和对齐方式，提升整体界面美观度

**实现细节**：
- 修改`fragment_rag_qa.xml`布局文件：
  - 为大模型Spinner添加`layout_marginEnd`和`layout_constraintEnd_toStartOf`约束
  - 添加思考模式CheckBox控件，使用ConstraintLayout进行精确定位
  - 调整控件间距，确保界面元素合理分布

#### 3.19.2 思考模式功能迁移

**功能迁移原因**：
- 思考模式是RAG问答过程中的核心功能，应该就近放置在使用场景附近
- 减少用户在设置页面和RAG界面之间的切换操作
- 提升功能的可发现性和使用便利性

**实现细节**：
1. **UI控件迁移**：
   - 在`RagQaFragment`中添加`checkBoxThinkingMode`控件
   - 从`SettingsFragment`中移除`switchNoThinking`相关代码
   - 更新布局文件，移除设置页面的思考模式开关

2. **状态管理逻辑**：
   - 实现复选框状态与`ConfigManager.getNoThinking()`的逻辑映射
   - 注意处理布尔值反转关系：`no_thinking=true`对应复选框未选中，`no_thinking=false`对应复选框选中
   - 在`loadConfig()`方法中加载思考模式状态并设置复选框初始状态

3. **配置同步机制**：
   - 添加复选框状态变化监听器，实时保存配置到文件
   - 使用`ConfigManager.setNoThinking()`方法保存用户选择
   - 确保配置变更立即生效，无需重启应用

**技术亮点**：
- 界面布局优化，提升视觉一致性和空间利用率
- 功能就近原则，将相关功能放在使用场景附近
- 配置状态实时同步，保证数据一致性
- 正确处理逻辑映射关系，避免状态混乱
- 代码清理和重构，移除冗余的UI控件和逻辑

#### 3.19.3 思考模式参数传递修复

**问题发现**：
在实际测试中发现，尽管思考模式复选框已正确迁移到RAG界面，但思考模式设置未能正确传递到推理过程中。具体表现为：
- 用户勾选思考模式复选框后，模型仍按非思考模式进行推理
- 生成的回答中包含思考内容，但采样参数未按思考模式调整
- 日志显示思考模式状态保存成功，但推理时未读取正确状态

**根本原因分析**：
1. **硬编码问题**：`LocalLlmHandler.inference()`方法中思考模式被硬编码为`false`
2. **参数传递缺失**：推理调用链中缺少从UI配置到底层推理的参数传递
3. **状态读取错误**：未从`ConfigManager`正确读取用户设置的思考模式状态

**修复实现**：

1. **修复LocalLlmHandler参数读取**：
```java
// 修复前：硬编码思考模式
boolean thinkingMode = false; // 默认关闭思考模式

// 修复后：从配置读取思考模式
boolean thinkingMode = !ConfigManager.getNoThinking(context);
LogManager.logD(TAG, "思考模式设置: " + (thinkingMode ? "启用" : "禁用"));
```

2. **停止推理逻辑优化**：
- 在`RagQaFragment.handleSendStopClick()`中添加停止标志重置逻辑
- 在`LocalLlmAdapter`中添加`resetStopFlag()`方法
- 确保每次开始新推理前重置停止状态，避免上次停止状态影响新推理

3. **推理循环停止检查**：
- `LocalLLMOnnxHandler.inferenceStream()`中已有完善的停止检查机制
- 在每个token生成循环中检查`handler.shouldStopInference()`
- 确保用户点击停止按钮后能及时中断推理过程

**修复效果验证**：
- 思考模式勾选后，推理参数正确调整（temperature=0.6, topK=20）
- 停止按钮功能正常，能及时中断推理过程
- 日志正确显示思考模式的启用/禁用状态
- 用户体验得到显著改善

#### 3.19.5 思考模式实现逻辑优化

**问题背景**：
原有的思考模式实现通过在模板中添加`<think>`标记来控制思考模式，但这种方式与模型的实际工作机制不符。正确的实现应该是：
- `enableThinking=true`：启用思考模式，不添加任何特殊指令
- `enableThinking=false`：禁用思考模式，在用户提示词尾部添加`/no_think`指令

**实现修改**：
在`HuggingfaceTokenizer.java`的`applyChatTemplate`方法中进行了以下修改：

1. **移除原有的思考标记添加逻辑**：
```java
// 原有逻辑（已移除）
if (enableThinking && !result.contains("<think>")) {
    // 添加<think>标记的代码
}
```

2. **新增/no_think指令添加逻辑**：
```java
// 新逻辑：禁用思考模式时添加/no_think指令
if (!enableThinking) {
    // 在最后一个用户消息尾部添加/no_think指令
    String userStart = imStart + userRole;
    int lastUserIndex = result.lastIndexOf(userStart);
    if (lastUserIndex >= 0) {
        int userEndIndex = result.indexOf(imEnd, lastUserIndex);
        if (userEndIndex >= 0) {
            result = result.substring(0, userEndIndex) + "\n/no_think" + result.substring(userEndIndex);
        }
    }
}
```

**技术优势**：
- 符合模型的原生工作机制
- 通过指令而非模板标记控制思考模式
- 更加灵活和可控的实现方式
- 减少了模板复杂性，提高了可维护性

**影响范围**：
- `HuggingfaceTokenizer.applyChatTemplate()`方法
- `applyTemplate()`私有方法
- `applyDefaultTemplate()`私有方法
- 所有使用思考模式的推理流程

**实现细节**：
- 在`applyTemplate()`方法中，将原来的思考标记添加逻辑替换为`/no_think`指令添加逻辑
- 在`applyDefaultTemplate()`方法中，移除了助手角色的思考标记添加代码
- 确保`/no_think`指令只在禁用思考模式时添加，避免重复添加
- 保持与现有配置管理系统的兼容性

#### 3.19.4 ONNX Runtime版本升级

**升级背景**：
针对用户反馈的GPU加速问题，特别是华为HarmonyOS设备上NNAPI和OpenCL支持不足的问题，我们进行了ONNX Runtime版本升级。

**版本变更**：
- **升级前**：ONNX Runtime 1.15.1
- **升级后**：ONNX Runtime 1.21.0

**升级原因**：
1. **NNAPI弃用**：Google已弃用NNAPI，ONNX Runtime 1.19+版本中NNAPI被标记为弃用状态
2. **兼容性改进**：新版本对Android设备的GPU支持更加完善
3. **性能优化**：包含更多算子支持和内存管理优化
4. **安全更新**：修复了已知的安全漏洞和稳定性问题

**技术实现**：
```gradle
// build.gradle 更新
// 升级前
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.15.1'

// 升级后
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.21.0'
```

**GPU加速替代方案**：
由于NNAPI已被弃用，提供以下替代优化方案：

1. **CPU多线程优化**：
   - 利用ARM CPU的多核性能
   - 优化线程池配置
   - 内存访问模式优化

2. **模型量化**：
   - 使用INT8量化减少计算负担
   - 动态量化提升推理速度
   - 保持模型精度的同时提升性能

3. **内存优化**：
   - 减少模型加载时的内存占用
   - 优化推理过程中的内存分配
   - 及时释放不必要的内存资源

**文档更新**：
同步更新了`GPU_TROUBLESHOOTING_GUIDE.md`，添加了：
- ONNX Runtime版本升级说明
- NNAPI弃用的影响分析
- 针对华为HarmonyOS设备的优化建议
- CPU模式下的性能调优方案

**预期效果**：
- 提升在新Android设备上的兼容性
- 改善推理性能和稳定性
- 为未来的GPU加速技术迁移做好准备
- 提供更好的错误处理和日志信息

### 3.20 Rust分词器集成

为了提高分词性能和跨平台一致性，我们实现了基于Rust的分词器库，并通过JNI集成到Android应用中。这种实现方式具有以下优势：

1. **高性能分词处理**：
   - 使用Rust实现的HuggingFace Tokenizers库，性能远优于Java实现
   - 支持多种分词算法，包括BPE、WordPiece、Unigram等
   - 通过JNI接口实现Java与Rust代码的无缝集成

2. **分词器架构设计**：
   - 采用三层架构设计：
     - Rust原生库（tokenizers-ffi）：实现核心分词功能
     - JNI绑定层（tokenizers-jni）：提供Java调用Rust的接口
     - Java包装类（RustTokenizer）：提供友好的Java API
   - 使用统一的JSON格式进行结果传递，确保数据一致性

3. **LLM模型集成**：
   - 在`LocalLLMOnnxHandler`中集成Rust分词器，替代原有的Java实现
   - 从模型配置中读取特殊token（BOS、EOS等），确保分词结果正确
   - 优化`tokenizeInput`方法，直接调用Rust分词器处理输入文本

4. **词汇表相关代码清理**：
   - 完全移除`TokenizerManager`中的词汇表加载和管理相关代码：
     - 删除`vocab`和`vocabReverse`字段及其初始化
     - 移除`loadVocabFromFile`和`loadVocabFromArray`方法
     - 移除`getVocabSize`方法及其调用
   - 更新`EmbeddingModelHandler`中的日志输出，移除对`getVocabSize()`的引用
   - 简化`TokenizerManager`类，使其专注于Rust分词器的管理和使用
   - 确保所有分词操作直接使用Rust分词器实现，不再依赖Java词汇表

5. **错误处理与资源管理**：
   - 实现完善的异常处理机制，捕获并记录分词过程中的错误
   - 使用`Closeable`接口确保分词器资源正确释放
   - 添加`finalize`方法作为资源释放的最后保障

6. **模型配置优化**：
   - 在`ModelConfig`类中添加`bosToken`和`eosToken`字段及相关方法
   - 在`loadModelConfig`方法中解析并设置特殊token值
   - 确保模型路径正确传递给分词器，支持从模型目录加载tokenizer.json

7. **构建系统集成**：
   - 在app模块的build.gradle中添加对tokenizers-jni库的依赖
   - 确保settings.gradle中包含tokenizers-jni模块
   - 支持多种CPU架构（arm64-v8a、armeabi-v7a、x86、x86_64）

8. **性能与内存优化**：
   - 移除Java词汇表相关代码后，显著减少了内存占用
   - 避免了大型词汇表（通常有5万至25万个token）在内存中的重复存储
   - 减少了应用启动时间，不再需要加载和解析词汇表文件
   - 简化了代码结构，提高了可维护性和可读性

9. **JNI接口实现与多架构支持**：
   - 正确实现符合JNI命名规范的函数，确保与Java代码的无缝集成
   - 函数名称遵循`Java_包名_类名_方法名`格式，例如`Java_com_starlocalrag_tokenizers_TokenizerJNI_loadTokenizerFromFile`
   - 支持多种CPU架构，包括：
     - arm64-v8a：适用于现代高端设备
     - armeabi-v7a：适用于旧版设备
     - x86：适用于模拟器和部分平板
     - x86_64：适用于高性能模拟器和部分设备
   - 在`.cargo/config.toml`中正确配置交叉编译目标，确保每个架构都能正确编译
   - 实现库文件的自动复制和部署，简化开发流程

10. **库加载机制优化**：
    - 改进`NativeLibraryLoader`类，支持多种加载策略：
      - 直接加载：先尝试使用`System.loadLibrary`直接加载
      - 资源加载：如果直接加载失败，尝试从资源中加载
      - 本地文件加载：如果资源加载失败，尝试从本地文件系统加载
    - 增强错误处理和日志记录，提供详细的调试信息
    - 使用标准Java日志替代Android特定日志，提高跨平台兼容性
    - 正确处理临时文件的创建和清理，避免资源泄漏

11. **模型输入格式化优化**：
    - 实现了类似 Python `apply_chat_template` 的功能，从 tokenizer.json 中获取聊天模板
    - 使用与原始 Python 实现相同的消息格式，如 `{"role": "user", "content": "..."}`
    - 支持多种模型的聊天模板格式：
      - Qwen3: `<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n`
      - Llama: `[INST] {message} [/INST]`
      - 其他模型: 根据 tokenizer.json 中的模板动态处理
    - 实现了智能回退机制，当无法解析模板时根据模型类型选择适当的格式
    - 改进了分词结果的一致性，确保与 Python 实现的分词结果一致
    - 解决了字符编码问题，避免了之前出现的乱码问题

12. **Qwen3 模型集成优化**：
    - 重构了聊天模板处理逻辑，实现了更灵活的模型支持
    - 从 tokenizer.json 文件中动态读取 chat_template，不再依赖硬编码的特殊标记
    - 优化了 `applyTokenizerTemplate` 方法，支持多种模型格式：
      - 优先使用 tokenizer.json 中的 chat_template
      - 根据模型类型提供合适的回退模板
      - 支持特殊标记的动态替换，如 `<|im_start|>`、`<|im_end|>` 等
    - 添加了 `replaceTemplateVariable` 辅助方法，支持多种变量格式：
      - 支持 `{{ var }}`, `{{var}}`, `{{ var}}`, `{{var }}` 等多种格式
      - 支持 `messages[0]["content"]`, `messages[0].content` 等复杂路径
      - 使用正则表达式处理复杂的模板变量
    - 简化了 `tokenizeInput` 方法，直接调用 `applyTokenizerTemplate` 处理用户输入
    - 增强了错误处理机制，提供详细的日志信息
    - 实现了与 Python 实现的完全兼容性，确保分词结果一致
    - 提高了代码可维护性，便于未来支持更多模型类型
    - 在 `ModelConfig` 类中添加了 `getTokenizerJsonPath` 方法，简化了路径处理
    - 完全移除了对模型类型的硬编码检测，提高了代码的通用性

13. **特殊 Token 动态获取优化**：
    - 重构了 `LocalLLMOnnxHandler` 中的特殊 token 处理逻辑，消除了硬编码的特殊 token
    - 增强了 `RustTokenizer` 类，添加了以下新方法：
      - `getSpecialTokens()`：获取所有特殊 token 的映射
      - `getSpecialTokenId(String tokenName)`：获取指定特殊 token 的 ID
      - `getSpecialTokenContent(String tokenName)`：获取指定特殊 token 的内容
      - `getTokenizerConfig()`：获取分词器的完整配置
      - `decode(int[] ids)`：将 token ID 数组解码为文本
    - 修改了 Rust FFI 层，添加了新的函数：
      - `get_tokenizer_config`：获取分词器配置，包括特殊 token 信息
      - `decode`：将 token ID 解码为文本
    - 优化了 Rust 代码中的特殊 token 检测逻辑：
      - 从词汇表中动态识别特殊 token，如 `im_start`、`im_end` 等
      - 构建完整的特殊 token 映射，包括 ID 和内容
      - 支持多种特殊 token 格式，如 `<token>` 和 `token` 格式
    - 改进了 `applyTokenizerTemplate` 方法，使用动态获取的特殊 token：
      - 从分词器配置中获取特殊 token，而不是硬编码
      - 正确处理不同模型的特殊 token 格式
      - 增强了错误处理，当特殊 token 不存在时提供合理的回退机制
    - 实现了完整的 JNI 接口，确保 Java 和 Rust 代码之间的无缝集成
    - 修复了 JNI 方法签名问题：
      - 将所有 JNI 方法的签名从 `extern "system"` 改为 `extern "C"`，确保方法能被正确导出
      - 在方法名中添加参数类型签名，如 `getTokenizerConfig__J`，其中 `__J` 表示方法接受一个 `long` 类型参数
      - 对于带字符串参数的方法，使用 `Ljava_lang_String_2` 表示 `String` 类型
      - 添加了 `decode` 方法的 JNI 实现，支持将 token ID 解码为文本
      - 确保 `getTokenizerConfig` 方法能被正确导出和调用
    - 增强了日志记录，便于调试和监控特殊 token 的使用情况
    - 这些优化使得代码更加灵活，能够适应不同模型的特殊 token 格式，提高了代码的可维护性和通用性

14. **模型输出处理优化**：
    - 重构了 `LocalLLMOnnxHandler` 中的模型输出处理逻辑，移除了硬编码的模拟输出
    - 实现了从模型输出张量中动态提取 token IDs 的逻辑：
      - 支持多种形状的输出张量（一维、二维和三维）
      - 根据张量形状和类型自动选择最适合的提取方式
      - 对二维张量 [batch_size, seq_len] 的正确处理
      - 对一维张量 [seq_len] 的正确处理
      - 为三维张量 [batch_size, seq_len, vocab_size] 预留处理逻辑
    - 使用 `RustTokenizer.decode()` 方法正确解码模型输出：
      - 将提取的 token IDs 传递给 Rust 分词器进行解码
      - 正确处理数据类型转换，如 long[] 到 int[] 的转换
    - 增强了日志记录和错误处理：
      - 记录张量形状、类型和处理过程的详细信息
      - 提供清晰的错误信息，如无法提取 token IDs 或解码失败
      - 在出错时提供合理的回退机制和用户友好的错误提示
    - 这些优化使得模型输出处理更加灵活和健壮，能够适应不同模型的输出格式，提高了代码的可维护性和通用性

15. **分词日志记录功能优化**：
    - 改进了 `escapeString` 方法，保留原始 Unicode 字符：
      - 修改了字符串转换逻辑，不再将所有非ASCII字符转换为转义序列
      - 只对控制字符（ASCII < 32 或 = 127）进行转义处理
      - 保留了对敏感 token 的处理逻辑
    - 优化了 `tokenizeInput` 方法中的日志记录：
      - 移除了使用 Unicode 转义序列的原始日志，避免日志中出现难以阅读的转义序列
      - 改进了可读性更强的日志格式，使用格式化字符串确保对齐
      - 添加了一个简洁的摘要日志，只显示前几个和后几个 token，便于快速查看
    - 日志输出优化：
      - 确保 Unicode 字符在日志中以可读文本形式显示，而不是转义序列
      - 日志输出更加清晰、结构化，便于阅读和调试
      - 提供了不同粒度的日志信息，既有详细的分词结果，也有简洁的摘要
    - 优化了模型输入格式：
      - 修改了 `applyTokenizerTemplate` 方法，构建更紧凑的输入格式
      - 从提示词中移除了“用户问题: ”前缀，使用更简洁的格式
      - 修改了 `RagQaFragment` 类中的 `buildPromptWithKnowledgeBase` 和 `buildPromptWithoutKnowledgeBase` 方法，移除了前缀
      - 确保在日志中保留完整信息，同时在提交给LLM的提示词中使用更简洁的格式
    - 这些改进大大提高了日志的可读性和调试效率，特别是在处理中文等非ASCII字符时

这种基于Rust的分词器实现显著提高了分词性能和准确性，特别是在处理复杂模型（如Qwen3、Llama3等）时，能够确保分词结果与训练时完全一致，从而提高模型推理质量。同时，由于移除了Java实现的分词器和相关词汇表加载逻辑，应用的内存占用和启动时间也得到了优化。

15. **分词器实现合并优化**：
    - 将 `RustTokenizerAdapter` 的功能合并到 `RustTokenizer` 中并重命名为 `HuggingfaceTokenizer`：
      - 实现了 `Model` 接口，保持与现有代码的兼容性
      - 添加了 `tokenizeToLongArray` 方法，返回与原 `HuggingfaceTokenizer.tokenize` 兼容的 `long[][]` 格式结果
      - 实现了对特殊token的完整支持，包括 `[CLS]`, `[SEP]`, `[UNK]`, `[PAD]`, `[MASK]` 等
      - 添加了词汇表加载和管理功能，支持从文件和数组加载词汇表
    - 实现了完整的聊天模板处理功能：
      - 添加了 `applyChatTemplate` 方法，支持处理聊天消息数组
      - 支持自定义聊天模板和默认模板
      - 支持思考模式和生成提示选项
      - 支持继续最后一条消息的功能
      - 实现了模板变量替换和特殊标记处理
    - 更新了 `TokenizerManager` 和 `LocalLLMOnnxHandler` 类：
      - 更新了所有引用，使用 `HuggingfaceTokenizer` 替代 `RustTokenizer` 和 `RustTokenizerAdapter`
      - 改进了错误处理和资源管理，确保分词器资源被正确释放
    - 完全移除了 `RustTokenizerAdapter` 类：
      - 将其功能整合到 `HuggingfaceTokenizer` 中
      - 简化了项目结构，减少了代码冗余
    - 扩展了 `TokenizerJNI` 类：
      - 添加了 `getVocab`、`decodeIds` 和 `destroyTokenizer` 等方法
      - 确保 Java 和 Rust 代码之间的无缝集成
    - 这些优化使得分词器实现更加统一和高效，减少了代码冗余，提高了性能和可维护性
    - 通过将功能集中在一个类中，简化了代码结构，使得维护和扩展更加容易

16. **分词器接口统一与优化**：
    - 统一了`EmbeddingModelHandler`和`LocalLLMOnnxHandler`中的分词器接口：
      - 创建了完整的`TokenizerInterface`接口，标准化分词器功能
      - 所有分词操作都通过`TokenizerManager`进行，确保一致性
      - 移除了Java层的token处理逻辑，将其委托给JNI层
    - 增强了特殊token的处理能力：
      - 添加了`getSpecialTokenId`、`getSpecialTokenContent`、`isSpecialToken`等方法
      - 实现了`loadModelSpecialTokens`方法，从分词器中加载特殊token
      - 增强了错误处理和日志记录，便于调试
    - 优化了`LocalLLMOnnxHandler`类的分词器使用：
      - 修改`tokenizeInput`方法，使用`TokenizerManager`替代直接使用`HuggingfaceTokenizer`
      - 优化资源释放机制，使用`resetManager`方法重置分词器
      - 增强了错误处理和日志记录，提高稳定性
    - 添加了`decodeIds`方法，支持将token ID解码回文本：
      - 在`TokenizerInterface`接口中定义了方法签名
      - 在`TokenizerManager`类中实现了方法逻辑
      - 确保了分词和解码操作的一致性
    - 这些优化提高了代码的一致性和可维护性：
      - 所有分词操作都通过统一的接口进行，减少了重复代码
      - 简化了错误处理和资源管理，提高了程序稳定性
      - 使得代码结构更加清晰，便于后续维护和扩展

17. **分词器加载优化**：
    - 重构了`TokenizerManager`类的词汇表和特殊token加载流程：
      - 移除了冗余的`loadVocabFromJson`、`loadSpecialTokensFromJson`、`loadModelSpecialTokens`和`loadVocabFromFile`方法
      - 添加了更高效的`syncSpecialTokensFromTokenizer`方法，直接从HuggingfaceTokenizer实例获取特殊token信息
      - 避免了重复加载词汇表和特殊token，减少内存使用和日志输出
      - 简化了初始化流程，提高了代码可维护性
    - 优化了`HuggingfaceTokenizer`类的词汇表加载：
      - 不再加载完整词汇表，只记录词汇表大小
      - 只加载关键特殊token（如CLS、SEP、UNK、PAD、MASK），不加载全部token
      - 减少了内存使用和不必要的日志输出

18. **分词器特殊标记处理优化**：
    - 重构了`HuggingfaceTokenizer`类的特殊标记处理机制：
      - 添加了`specialTokenIds`和`preservedTokenIds`数组，分别存储需要过滤和需要保留的特殊标记ID
      - 实现了`loadSpecialTokenIds`方法，从 tokenizer.json 文件中动态加载特殊标记ID
      - 优化了`isSpecialTokenId`方法，直接考虑需要保留的特殊标记，简化了过滤逻辑
      - 添加了`decodeForModelOutput`方法，专门用于模型输出处理，过滤掉特殊标记但保留思考链标记
    - 修改了`TokenizerManager`类的`decodeIds`方法，使其调用`decodeForModelOutput`方法：
      - 确保所有模型输出都经过特殊标记过滤
      - 统一了解码处理逻辑，提高了代码可维护性
    - 这些优化解决了以下问题：
      - 避免了硬编码特殊标记ID，提高了代码的灵活性和适应性
      - 正确处理了各种特殊标记格式（如`<|im_start|>`、`<im_start|>`等）
      - 确保了思考链相关的标记（`<think>`和`</think>`）被正确保留
      - 提高了解码过程的稳定性和可靠性

19. **本地LLM流式输出功能实现**：
    - 添加了`StreamingCallback`接口，用于处理流式输出：
      - `onToken`方法：接收生成的单个token
      - `onComplete`方法：标记生成完成
      - `onError`方法：处理生成过程中的错误
    - 修改了`inference`方法，使其调用新的`inferenceStream`方法：
      - 使用`CountDownLatch`同步机制等待流式推理完成
      - 收集流式生成的所有token，组合成完整响应
      - 优化了错误处理，将`OrtException`转换为更通用的`RuntimeException`
    - 实现了`inferenceStream`方法，支持实时token生成：
      - 在单独线程中执行推理过程，避免阻塞UI线程
      - 使用`StreamingCallback`将生成的token实时发送回UI
      - 实现了完整的错误处理和资源释放机制
    - 优化了批量解码功能：
      - 比较逐个解码和批量解码的结果，选择最优解码方式
      - 添加了详细的日志记录，便于调试和分析
      - 实现了异常处理机制，在批量解码失败时回退到逐个解码
    - 优化了模板处理逻辑：
      - 修改`applyTokenizerTemplate`方法，使用JNI层的`applyChatTemplate`方法处理模板
      - 移除Java层硬编码的模板格式，将模板处理逻辑完全交由JNI层负责
      - 在出错时直接抛出异常，避免使用不一致的回退模板
      - 保持代码层次清晰，提高可维护性和一致性
    - 实现了本地LLM与现有UI的集成：
      - 修改了`LocalLlmAdapter`类中的`executeInference`方法，模拟在线模型的行为
      - 添加了模型回答标题行，确保与在线模型的输出格式一致
      - 增强了日志记录，便于跟踪token的生成和传递过程
      - 确保每个token都正确地传递给UI，实现实时显示
    - 这些改进使本地LLM能够提供与在线模型类似的流式输出体验，显著提升了用户交互体验：
      - 用户可以实时看到模型生成的内容，无需等待完整响应
      - 应用保持响应性，即使在处理长文本生成时
      - 统一了本地和在线模型的用户体验，提供一致的交互模式
    - 流式输出模块的设计考虑了以下因素：
      - **层次化的回调机制**：从底层的`LocalLLMOnnxHandler`到中间层的`LocalLlmHandler`再到适配层的`LocalLlmAdapter`，最终到UI层的`RagQaFragment`
      - **一致的接口设计**：确保本地和在线模型使用相同的接口，便于统一处理
      - **完善的错误处理**：在每一层都实现了错误处理机制，确保异常情况下的稳定性
      - **统一的模板处理**：将模板处理逻辑统一放在JNI层，避免在Java层重复实现，减少代码冗余

20. **本地LLM推理统计功能实现**：
    - **增强的推理性能统计报告**：在模型输出完成后自动追加详细的性能统计信息
      - **Token生成统计**：
        - 本次生成token数量：记录当前推理会话生成的token总数
        - 累计生成token数量：记录应用启动以来的总token生成数
        - 推理耗时：精确记录从开始推理到结束的总时间（秒）
        - 生成速率：计算每秒生成的token数量（tokens/秒）
      - **内存使用统计**：
        - JVM最大可用内存：显示Java虚拟机的最大内存限制
        - JVM当前使用内存：显示当前应用实际使用的内存
        - 推理前应用内存：记录推理开始前的内存基线
        - 推理期间最大内存：监控推理过程中的内存峰值
        - LLM推理消耗内存：计算推理过程中的额外内存占用
      - **系统资源统计**：
        - 系统总内存：显示设备的总内存容量
        - 系统可用内存：显示当前系统可用内存
        - 内存使用率：计算系统整体内存使用百分比
      - **推理配置信息**：
        - 引擎类型：显示使用的推理引擎（ONNX Runtime GenAI）
        - API模式：显示使用的API类型（高级API/低级API）
        - GPU加速状态：显示是否启用GPU加速
        - 设备类型：显示当前设备类型
        - KV缓存大小：显示配置的KV缓存token数量
        - 推理线程数：显示CPU推理时使用的线程数

21. **GPU加速优化与HarmonyOS适配**：
    - **HarmonyOS GPU适配最佳实践**：
      - 针对华为HarmonyOS设备实现专门的GPU优化策略，优先使用OpenCL和Vulkan API
      - 实现HarmonyOS特定的GPU权限检测，包括ohos.permission.USE_GPU权限验证
      - 提供系统级GPU调度策略优化，包括GPU频率调节和内存管理
      - 支持Vulkan计算支持检测和设备本地内存配置优化
    - **Mali GPU性能优化策略**：
      - 实现Mali GPU架构自动检测，支持Mali-G610等主流Mali GPU
      - 提供Mali GPU特定的ONNX Runtime配置，包括线程数、内存带宽、卷积优化
      - 实现Mali GPU性能实时监控，包括GPU使用率、频率和温度监控
      - 支持GPU频率调节器优化，自动设置性能模式和最小频率
      - 提供Mali GPU内存分配策略，包括内存池大小、压缩和垃圾回收优化
    - **OpenGL ES计算着色器加速**：
      - 实现OpenGL ES 3.2计算着色器的GPU加速支持
      - 提供基础计算和矩阵乘法的计算着色器程序
      - 实现GPU内存缓冲区管理和计算结果高效读取
      - 支持计算着色器资源的自动清理和释放
    - **GPU加速错误处理与降级策略**：
      - 实现渐进式GPU加速启用，按优先级尝试不同GPU后端
      - 提供详细的GPU错误诊断和针对性建议
      - 实现GPU加速失败时的自动CPU降级机制
      - 支持Mali GPU特性检测，包括OpenCL、Vulkan、计算着色器和FP16支持检查
    - **GPU配置管理与权限处理**：
      - 在AndroidManifest.xml中添加完整的GPU相关权限和硬件特性声明
      - 实现GPU配置有效性检查，包括硬件特性、权限和驱动支持
      - 提供GPU故障排除指南，包含常见问题FAQ和修复方案
      - 支持HarmonyOS和标准Android设备的差异化GPU配置策略
      - 生成速度：计算每秒生成的token数量（token/s）
    - 实现了Android兼容的内存监控机制：
      - **Android兼容性改进**：移除了JVM特定的`java.lang.management`包依赖，改用Android原生API
      - 使用`Runtime.getRuntime()`获取应用内存使用情况，兼容Android环境
      - 集成`ActivityManager`和`MemoryInfo`获取系统级内存信息
      - 实现`startMemoryMonitoring()`、`updateMemoryMonitoring()`和`getMemoryStats()`方法
      - 在推理过程中实时更新内存峰值，记录推理前后的内存变化
      - 提供详细的内存统计报告，包括应用内存和系统内存状态
    - 优化了统计数据的收集和计算：
      - 精确记录推理开始和结束时间
      - 统计生成的总token数量
      - 计算每秒生成的token数（token/s）
      - 使用初始内存基线，计算推理过程中的额外内存占用
    - 统计信息展示设计：
      - 使用分隔线与正常输出内容区分
      - 采用清晰的格式化布局，便于用户阅读
      - 统计信息包括：推理时间、内存占用和生成速度
      - 自动添加到输出结果尾部，无需用户额外操作
    - **统一内存监控机制**：
      - 统一了`LocalLLMOnnxHandler`和`LocalLLMOnnxRuntimeGenAIHandler`的内存监控实现
      - 使用统一的内存差值估算方法：`memoryMaxDuringInference - memoryBeforeInference`
      - 实现了`startMemoryMonitoring()`、`updateMemoryMonitoring()`和`getMemoryStats()`方法
      - 提供一致的LLM推理内存消耗统计和回调打印
      - 在推理过程中实时监控内存使用峰值，准确估算LLM推理消耗
    - **OnnxRuntimeGenAI模型保持机制**：
      - 实现模型加载后的持久化保持，避免不必要的重复加载
      - 添加`keepModelLoaded`标志和`currentModelPath`跟踪当前加载的模型
      - 支持模型重用：相同模型路径时直接重用已加载的模型实例
      - 提供`forceRelease()`方法用于强制释放资源（模型切换或应用退出时）
      - 实现应用生命周期事件处理：`onLifecycleEvent()`方法
      - 支持检测模型是否被系统回收：`isModelRecycled()`方法
      - 模型保持条件：a) 用户未切换模型 b) 模型未被OS优化回收（如熄屏后）
    - **华为设备启动优化**：
      - 修复华为手机启动无响应问题：在`MainActivity.performGPUConfigCheck()`中检测华为设备并跳过GPU配置检查
      - 优化`GPUErrorHandler.handleHuaweiSpecificIssues()`：移除可能导致启动卡顿的反射调用，只保留轻量级系统属性设置
      - 在`RagQaFragment`中为华为设备提供更保守的UI更新参数：`MIN_CHAR_CHANGE=30`，`UPDATE_INTERVAL=300ms`
      - 将GPU配置检查移至后台线程执行，避免阻塞主线程导致启动无响应
      - 华为设备检测范围：华为(huawei)、荣耀(honor)品牌设备
    - **性能统计报告优化**：
      - 简化统计报告输出格式：将复杂的多行emoji装饰格式改为简洁的单行格式
      - 统一格式：`tokens计数: XX • 耗时: XXXs • 速率: XXX token/s • JVM内存最大消耗: XXXMB • LLM内存最大消耗: XXXXMB • 系统最大内存消耗: XXXXMB`
      - 修复内存监控问题：提高监控频率从500ms到200ms，增强内存峰值捕获精度
      - 优化内存基线测量：推理前强制垃圾回收，确保准确的内存基线
      - 增加详细的内存监控调试信息，便于诊断内存统计异常问题
    - 这些功能为用户和开发者提供了重要的性能指标：
      - 帮助用户了解模型在其设备上的实际性能
      - 便于开发者进行性能调优和问题诊断
      - 提供了比较不同模型性能的客观数据
      - 有助于识别性能瓶颈和优化方向
      - 通过模型保持机制显著提升多轮对话的响应速度
