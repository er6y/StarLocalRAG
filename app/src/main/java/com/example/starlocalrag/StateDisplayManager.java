package com.example.starlocalrag;

import android.content.Context;
import com.example.starlocalrag.R;

/**
 * State Display Manager
 * Responsible for converting state constants to display text, supports multiple languages
 */
public class StateDisplayManager {
    
    private final Context context;
    
    public StateDisplayManager(Context context) {
        this.context = context;
    }
    
    /**
     * Get display text for model state
     */
    public String getModelStateDisplay(String state) {
        switch (state) {
            case AppConstants.ModelState.DOWNLOADING:
                return context.getString(R.string.common_download) + "中";
            case AppConstants.ModelState.DOWNLOADED:
                return "已" + context.getString(R.string.common_download);
            case AppConstants.ModelState.NOT_DOWNLOADED:
                return "未" + context.getString(R.string.common_download);
            case AppConstants.ModelState.DOWNLOAD_FAILED:
                return context.getString(R.string.common_download) + context.getString(R.string.common_failed);
            case AppConstants.ModelState.LOADING:
                return "加载中";
            case AppConstants.ModelState.LOADED:
                return "已加载";
            case AppConstants.ModelState.LOAD_FAILED:
                return "加载" + context.getString(R.string.common_failed);
            case AppConstants.ModelState.UNLOADED:
                return "未加载";
            // MODEL_STATUS constants
            case AppConstants.MODEL_STATUS_DIRECTORY_NOT_EXIST:
                return "目录不存在";
            case AppConstants.MODEL_STATUS_NOT_FOUND:
                return "未找到";
            case AppConstants.MODEL_STATUS_NO_AVAILABLE:
                return "无可用";
            case AppConstants.MODEL_STATUS_FETCH_FAILED:
                return "获取" + context.getString(R.string.common_failed);
            case AppConstants.MODEL_STATUS_PLEASE_SELECT_EMBEDDING:
                return "请选择嵌入模型";
            case AppConstants.MODEL_STATUS_PARSE_FAILED:
                return "解析" + context.getString(R.string.common_failed);
            default:
                return context.getString(R.string.common_unknown_model);
        }
    }
    
    /**
     * Get display text for knowledge base state
     */
    public String getKnowledgeBaseStateDisplay(String state) {
        switch (state) {
            case AppConstants.KnowledgeBaseState.BUILDING:
                return context.getString(R.string.kb_state_building);
            case AppConstants.KnowledgeBaseState.BUILT:
                return context.getString(R.string.kb_state_built);
            case AppConstants.KnowledgeBaseState.NOT_BUILT:
                return context.getString(R.string.kb_state_not_built);
            case AppConstants.KnowledgeBaseState.BUILD_FAILED:
                return context.getString(R.string.common_build_failed);
            case AppConstants.KnowledgeBaseState.LOADING:
                return context.getString(R.string.common_loading);
            case AppConstants.KnowledgeBaseState.LOADED:
                return context.getString(R.string.common_loaded);
            case AppConstants.KnowledgeBaseState.LOAD_FAILED:
                return context.getString(R.string.common_failed);
            case AppConstants.KnowledgeBaseState.EMPTY:
                return context.getString(R.string.kb_state_empty);
            default:
                return context.getString(R.string.common_unknown_state);
        }
    }
    
    /**
     * Get display text for progress state
     */
    public String getProgressStateDisplay(String state) {
        switch (state) {
            case AppConstants.ProgressState.INITIALIZING:
                return "初始化中";
            case AppConstants.ProgressState.PROCESSING:
                return "处理中";
            case AppConstants.ProgressState.COMPLETED:
                return context.getString(R.string.common_completed);
            case AppConstants.ProgressState.FAILED:
                return context.getString(R.string.common_failed);
            case AppConstants.ProgressState.CANCELLED:
                return context.getString(R.string.common_cancelled);
            case AppConstants.ProgressState.PAUSED:
                return context.getString(R.string.common_pause) + "中";
            default:
                return "未知状态";
        }
    }
    
    /**
     * Get display text for API URL
     */
    public String getApiUrlDisplay(String apiUrl) {
        switch (apiUrl) {
            case AppConstants.ApiUrl.LOCAL:
                return "本地";
            case AppConstants.ApiUrl.OPENAI:
                return "OpenAI";
            case AppConstants.ApiUrl.CUSTOM:
                return "自定义";
            case AppConstants.ApiUrl.NEW:
                return "新建";
            default:
                return apiUrl; // 直接返回原始值
        }
    }
    
    /**
     * Get display text for embedding model
     */
    public String getEmbeddingModelDisplay(String model) {
        switch (model) {
            case AppConstants.EmbeddingModel.BGE_SMALL:
                return context.getString(R.string.embedding_model_bge_small);
            case AppConstants.EmbeddingModel.BGE_BASE:
                return context.getString(R.string.embedding_model_bge_base);
            case AppConstants.EmbeddingModel.BGE_LARGE:
                return context.getString(R.string.embedding_model_bge_large);
            case AppConstants.EmbeddingModel.SENTENCE_TRANSFORMER:
                return context.getString(R.string.embedding_model_sentence_transformer);
            case AppConstants.EmbeddingModel.ROOT_DIRECTORY:
                return context.getString(R.string.embedding_model_root_directory);
            default:
                return context.getString(R.string.common_unknown_model);
        }
    }
    
    /**
     * Get display text for reranker model
     */
    public String getRerankerModelDisplay(String model) {
        switch (model) {
            case AppConstants.RerankerModel.NONE:
                return context.getString(R.string.common_none);
            case AppConstants.RerankerModel.BGE_RERANKER:
                return context.getString(R.string.reranker_model_bge_reranker);
            case AppConstants.RerankerModel.CUSTOM:
                return context.getString(R.string.common_custom);
            default:
                return context.getString(R.string.reranker_model_unknown);
        }
    }
    
    /**
     * Get display text for menu item
     */
    public String getMenuItemDisplay(String menuItem) {
        switch (menuItem) {
            case AppConstants.MenuItem.COPY:
                return context.getString(R.string.common_copy);
            case AppConstants.MenuItem.SELECT_ALL:
                return context.getString(R.string.common_select_all);
            case AppConstants.MenuItem.CLEAR:
                return context.getString(R.string.common_clear);
            case AppConstants.MenuItem.SAVE:
                return context.getString(R.string.common_save);
            case AppConstants.MenuItem.SHARE:
                return context.getString(R.string.common_share);
            default:
                return context.getString(R.string.common_unknown_operation);
        }
    }
    
    /**
     * Get display text for file type
     */
    public String getFileTypeDisplay(String fileType) {
        switch (fileType) {
            case AppConstants.FileType.PDF:
                return context.getString(R.string.file_type_pdf);
            case AppConstants.FileType.TXT:
                return context.getString(R.string.file_type_txt);
            case AppConstants.FileType.DOC:
                return context.getString(R.string.file_type_doc);
            case AppConstants.FileType.DOCX:
                return context.getString(R.string.file_type_docx);
            case AppConstants.FileType.MD:
                return context.getString(R.string.file_type_md);
            case AppConstants.FileType.HTML:
                return context.getString(R.string.file_type_html);
            default:
                return context.getString(R.string.common_unknown_file_type);
        }
    }
    
    /**
     * Get display text for validation messages
     */
    public String getValidationDisplay(String validationKey) {
        switch (validationKey) {
            case AppConstants.VALIDATION_PLEASE_SELECT_VALID_EMBEDDING:
                return context.getString(R.string.validation_please_select_valid_embedding);
            case AppConstants.VALIDATION_PLEASE_SELECT_FILES:
                return context.getString(R.string.validation_please_select_files);
            case AppConstants.VALIDATION_KB_NAME_CANNOT_BE_EMPTY:
                return context.getString(R.string.validation_kb_name_cannot_be_empty);
            default:
                return context.getString(R.string.common_unknown_validation);
        }
    }
    
    /**
     * Get display text for dialog
     */
    public String getDialogDisplay(String dialogKey) {
        switch (dialogKey) {
            case AppConstants.DIALOG_TITLE_CONFIRM_INTERRUPT:
                return context.getString(R.string.dialog_title_confirm_interrupt);
            case AppConstants.DIALOG_MESSAGE_CONFIRM_INTERRUPT:
                return context.getString(R.string.dialog_message_confirm_interrupt);
            case AppConstants.DIALOG_TITLE_KB_EXISTS:
                return context.getString(R.string.dialog_title_kb_exists);
            case AppConstants.DIALOG_MESSAGE_KB_EXISTS:
                return context.getString(R.string.dialog_message_kb_exists);
            case AppConstants.DIALOG_TITLE_NEED_FULL_FILE_ACCESS:
                return context.getString(R.string.dialog_title_need_full_file_access);
            case AppConstants.DIALOG_MESSAGE_NEED_FULL_FILE_ACCESS:
                return context.getString(R.string.dialog_message_need_full_file_access);
            case AppConstants.DIALOG_TITLE_NEED_STORAGE_PERMISSION:
                return context.getString(R.string.dialog_title_need_storage_permission);
            case AppConstants.DIALOG_MESSAGE_NEED_STORAGE_PERMISSION:
                return context.getString(R.string.dialog_message_need_storage_permission);
            case AppConstants.DIALOG_TITLE_ABOUT:
                return context.getString(R.string.dialog_title_about);
            case AppConstants.DIALOG_MESSAGE_ABOUT:
                return "StarLocalRAG - 本地RAG问答系统";
            default:
                return dialogKey; // Return original key as fallback
        }
    }

    /**
     * Get display text for button
     */
    public String getButtonDisplay(String buttonKey) {
        switch (buttonKey) {
            case AppConstants.BUTTON_TEXT_OK:
                return context.getString(R.string.common_ok);
            case AppConstants.BUTTON_TEXT_CANCEL:
                return context.getString(R.string.common_cancel);
            case AppConstants.BUTTON_TEXT_CREATE_KB:
                return context.getString(R.string.button_create_kb);
            case AppConstants.BUTTON_TEXT_CONTINUE:
                return context.getString(R.string.common_continue);
            case AppConstants.BUTTON_TEXT_NEW_KB:
                return context.getString(R.string.button_new_kb);
            case AppConstants.BUTTON_TEXT_OVERWRITE:
                return context.getString(R.string.common_overwrite);
            case AppConstants.BUTTON_TEXT_APPEND:
                return context.getString(R.string.common_append);
            case AppConstants.BUTTON_TEXT_GO_TO_SETTINGS:
                return context.getString(R.string.button_go_to_settings);
            default:
                return buttonKey; // Return original key as fallback
        }
    }

    /**
     * Automatically select corresponding display method based on state key type
     */
    public String getDisplayText(String stateType, String state) {
        switch (stateType) {
            case "model":
                return getModelStateDisplay(state);
            case "knowledge_base":
                return getKnowledgeBaseStateDisplay(state);
            case "progress":
                return getProgressStateDisplay(state);
            case "api_url":
                return getApiUrlDisplay(state);
            case "embedding_model":
                return getEmbeddingModelDisplay(state);
            case "menu_item":
                return getMenuItemDisplay(state);
            case "file_type":
                return getFileTypeDisplay(state);
            case "validation":
                return getValidationDisplay(state);
            case "button":
                return getButtonDisplay(state);
            default:
                return state; // Return original state value as fallback
        }
    }
    
    // Backward compatible methods, maintain compatibility with old versions
    public String getModelStateDisplayText(String stateKey) {
        return getModelStateDisplay(stateKey);
    }
    
    public String getKnowledgeBaseStateDisplayText(String stateKey) {
        return getKnowledgeBaseStateDisplay(stateKey);
    }
    
    public String getProgressStateDisplayText(String stateKey) {
        return getProgressStateDisplay(stateKey);
    }
    
    public String getApiUrlDisplayText(String stateKey) {
        return getApiUrlDisplay(stateKey);
    }
    
    // Static methods for backward compatibility
    public static String getApiUrlDisplayText(Context context, String apiUrl) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.getApiUrlDisplay(apiUrl);
    }
    
    public static String getKnowledgeBaseStatusDisplayText(Context context, String status) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.getKnowledgeBaseStateDisplay(status);
    }
    
    public static boolean isModelStatusDisplayText(Context context, String status) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.isModelStatus(status);
    }
    
    public static boolean isKnowledgeBaseStatusDisplayText(Context context, String status) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.isKnowledgeBaseStatus(status);
    }
    
    /**
     * Check if it is a model status
     */
    public boolean isModelStatus(String status) {
        return AppConstants.ModelState.LOADING.equals(status) ||
               AppConstants.ModelState.LOADED.equals(status) ||
               AppConstants.ModelState.NOT_DOWNLOADED.equals(status);
    }
    
    /**
     * Check if it is a knowledge base status
     */
    public boolean isKnowledgeBaseStatus(String status) {
        return AppConstants.KnowledgeBaseState.LOADING.equals(status) ||
               AppConstants.KnowledgeBaseState.EMPTY.equals(status) ||
               AppConstants.KnowledgeBaseState.READY.equals(status);
    }
    
    public static String getDialogDisplayText(Context context, String key) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.getDisplayText("dialog", key);
    }
    
    public static String getButtonDisplayText(Context context, String key) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.getDisplayText("button", key);
    }
    
    public static String getProcessingStatusDisplayText(Context context, String key) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.getDisplayText("processing_status", key);
    }
    
    public static String getErrorDisplayText(Context context, String key) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.getDisplayText("error", key);
    }
    
    public static String getRerankerModelDisplayText(Context context, String key) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.getDisplayText("reranker_model", key);
    }
    
    public static boolean isProcessingStatusDisplayText(Context context, String status, String targetStatus) {
        return status != null && status.equals(targetStatus);
    }
    
    public static String getModelStatusDisplayText(Context context, String status) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.getModelStateDisplay(status);
    }

    public static String getSuccessDisplayText(Context context, String successKey) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.getDisplayText("success", successKey);
    }

    public static String getValidationDisplayText(Context context, String validationKey) {
        StateDisplayManager manager = new StateDisplayManager(context);
        return manager.getDisplayText("validation", validationKey);
    }

    /**
     * Get display text for menu items
     */
    public String getMenuDisplay(String menuKey) {
        switch (menuKey) {
            case AppConstants.MENU_SWITCH_TO_ENGLISH:
                return context.getString(R.string.menu_switch_to_english);
            case AppConstants.MENU_SWITCH_TO_CHINESE:
                return context.getString(R.string.menu_switch_to_chinese);
            case AppConstants.MENU_CONVERT_TO_NOTE:
                return context.getString(R.string.menu_convert_to_note);
            default:
                return menuKey;
        }
    }

}