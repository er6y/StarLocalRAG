package com.example.starlocalrag;

import android.content.Context;
import android.util.Log;
import android.util.TypedValue;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.widget.TextView;
import android.widget.Toast;

/**
 * 文本视图缩放辅助类
 * 提供两指缩放功能，可以调整文本视图的字体大小
 * 同时保留文本选择、长按菜单和滚动功能
 */
public class TextViewZoomHelper {
    private static final String TAG = "TextViewZoomHelper";
    
    // 默认字体大小范围（单位：sp）
    public static final float MIN_TEXT_SIZE = 10f;
    public static final float MAX_TEXT_SIZE = 24f;
    
    // 缩放灵敏度调整因子，值越小，缩放越不敏感
    private static final float SCALE_SENSITIVITY = 0.2f; // 降低灵敏度
    
    private final TextView textView;
    private final ScaleGestureDetector scaleGestureDetector;
    private float currentTextSize;
    private final Context context;
    private boolean isScaling = false;
    private long lastScaleTime = 0;
    private static final long SCALE_COOLDOWN = 500; // 增加冷却时间
    private float lastNotifiedTextSize = 0; // 上次通知的字体大小
    private static final float TEXT_SIZE_NOTIFICATION_THRESHOLD = 0.5f; // 字体大小变化阈值，超过此值才显示Toast
    private long lastToastTime = 0; // 上次显示Toast的时间
    private static final long TOAST_COOLDOWN = 1000; // Toast显示冷却时间（毫秒）
    private long lastTextSizeUpdateTime = 0; // 上次更新字体大小的时间
    private static final long TEXT_SIZE_UPDATE_INTERVAL = 100; // 字体大小更新间隔（毫秒）
    
    /**
     * 构造函数
     * @param context 上下文
     * @param textView 要添加缩放功能的文本视图
     */
    public TextViewZoomHelper(Context context, TextView textView) {
        this.context = context;
        this.textView = textView;
        
        // 获取当前字体大小（单位：sp）
        currentTextSize = textView.getTextSize() / context.getResources().getDisplayMetrics().scaledDensity;
        lastNotifiedTextSize = currentTextSize;
        
        // 创建缩放手势检测器
        scaleGestureDetector = new ScaleGestureDetector(context, new ScaleListener());
        
        // 设置触摸监听器
        textView.setOnTouchListener((v, event) -> {
            try {
                // 处理多点触控事件
                int pointerCount = event.getPointerCount();
                
                // 检测事件类型
                int action = event.getActionMasked();
                
                // 如果是单点触控且不是缩放状态，先让TextView处理（允许文本选择）
                if (pointerCount == 1 && !isScaling && 
                    System.currentTimeMillis() - lastScaleTime > SCALE_COOLDOWN) {
                    // 允许TextView处理单点触控事件（文本选择、滚动等）
                    v.performClick(); // 添加这行以支持辅助功能
                    boolean handled = textView.onTouchEvent(event);
                    
                    // 如果是ACTION_DOWN或ACTION_UP，不要消费事件，让其他监听器也能处理
                    if (action == MotionEvent.ACTION_DOWN || action == MotionEvent.ACTION_UP) {
                        return false;
                    }
                    
                    // 如果TextView已处理该事件，则返回true
                    if (handled) {
                        return true;
                    }
                }
                
                // 处理缩放手势
                boolean handled = scaleGestureDetector.onTouchEvent(event);
                
                // 如果是多点触控或者正在缩放，拦截事件
                if (pointerCount >= 2 || isScaling) {
                    // 如果缩放结束，设置标志并记录时间
                    if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL || 
                        action == MotionEvent.ACTION_POINTER_UP) {
                        isScaling = false;
                        lastScaleTime = System.currentTimeMillis();
                    }
                    return true;
                }
                
                return handled;
            } catch (Exception e) {
                // 捕获并记录所有触摸事件处理异常
                Log.e(TAG, "处理触摸事件时发生异常: " + e.getMessage(), e);
                return false;
            }
        });
        
        // 初始化时不显示字体大小信息，避免干扰用户
    }
    
    /**
     * 缩放监听器
     */
    private class ScaleListener extends ScaleGestureDetector.SimpleOnScaleGestureListener {
        @Override
        public boolean onScaleBegin(ScaleGestureDetector detector) {
            isScaling = true;
            return true;
        }
        
        @Override
        public boolean onScale(ScaleGestureDetector detector) {
            try {
                // 计算新的字体大小，应用灵敏度调整
                float scaleFactor = 1.0f + (detector.getScaleFactor() - 1.0f) * SCALE_SENSITIVITY;
                float newTextSize = currentTextSize * scaleFactor;
                
                // 限制字体大小在最小值和最大值之间
                if (newTextSize < MIN_TEXT_SIZE) {
                    newTextSize = MIN_TEXT_SIZE;
                } else if (newTextSize > MAX_TEXT_SIZE) {
                    newTextSize = MAX_TEXT_SIZE;
                }
                
                // 如果字体大小有明显变化，更新文本视图
                if (Math.abs(newTextSize - currentTextSize) > 0.1f && 
                    System.currentTimeMillis() - lastTextSizeUpdateTime >= TEXT_SIZE_UPDATE_INTERVAL) {
                    
                    // 更新当前字体大小
                    currentTextSize = newTextSize;
                    
                    // 更新TextView的字体大小（单位：sp）
                    textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, currentTextSize);
                    
                    // 记录更新时间
                    lastTextSizeUpdateTime = System.currentTimeMillis();
                    
                    // 如果字体大小变化超过阈值，显示Toast提示
                    if (Math.abs(currentTextSize - lastNotifiedTextSize) >= TEXT_SIZE_NOTIFICATION_THRESHOLD &&
                        System.currentTimeMillis() - lastToastTime >= TOAST_COOLDOWN) {
                        
                        // 更新上次通知的字体大小
                        lastNotifiedTextSize = currentTextSize;
                        
                        // 显示Toast提示
                        String message = String.format("字体大小: %.1fsp (范围: %.1f-%.1fsp)", 
                                currentTextSize, MIN_TEXT_SIZE, MAX_TEXT_SIZE);
                        Toast.makeText(context, message, Toast.LENGTH_SHORT).show();
                        
                        // 记录Toast显示时间
                        lastToastTime = System.currentTimeMillis();
                    }
                    
                    // 保存字体大小到配置
                    saveTextSize();
                }
                
                return true;
            } catch (Exception e) {
                // 捕获并记录所有缩放处理异常
                Log.e(TAG, "处理缩放事件时发生异常: " + e.getMessage(), e);
                return false;
            }
        }
        
        @Override
        public void onScaleEnd(ScaleGestureDetector detector) {
            isScaling = false;
            lastScaleTime = System.currentTimeMillis();
            
            // 缩放结束时显示最终字体大小
            if (Math.abs(currentTextSize - lastNotifiedTextSize) >= 0.1f) {
                showCurrentTextSize();
                lastNotifiedTextSize = currentTextSize;
                lastToastTime = System.currentTimeMillis();
            }
        }
    }
    
    /**
     * 保存字体大小到配置
     */
    private void saveTextSize() {
        // 根据文本视图的ID保存不同的配置键
        String configKey = getConfigKeyForTextView();
        if (configKey != null) {
            ConfigManager.setFloat(context, configKey, currentTextSize);
        }
    }
    
    /**
     * 根据文本视图的ID获取配置键
     * @return 配置键
     */
    private String getConfigKeyForTextView() {
        int id = textView.getId();
        
        if (id == R.id.textViewResponse) {
            return ConfigManager.KEY_RAG_RESPONSE_TEXT_SIZE;
        } else if (id == R.id.textViewFileList) {
            return ConfigManager.KEY_BUILD_SELECTED_FILES_TEXT_SIZE;
        } else if (id == R.id.textViewProgress) {
            return ConfigManager.KEY_BUILD_PROGRESS_TEXT_SIZE;
        } else if (id == R.id.editTextContent) {
            return ConfigManager.KEY_NOTE_CONTENT_TEXT_SIZE;
        } else if (id == R.id.textViewLog) {
            return ConfigManager.KEY_LOG_CONTENT_TEXT_SIZE;
        }
        
        return null;
    }
    
    /**
     * 显示当前字体大小
     */
    private void showCurrentTextSize() {
        String message = String.format("字体大小: %.1fsp (范围: %.1f-%.1fsp)", 
                currentTextSize, MIN_TEXT_SIZE, MAX_TEXT_SIZE);
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show();
    }
    
    /**
     * 加载保存的字体大小
     */
    public void loadSavedTextSize() {
        String configKey = getConfigKeyForTextView();
        if (configKey != null) {
            // 获取保存的字体大小，如果没有则使用当前大小
            float savedSize = ConfigManager.getFloat(context, configKey, currentTextSize);
            
            // 限制字体大小在最小值和最大值之间
            if (savedSize < MIN_TEXT_SIZE) {
                savedSize = MIN_TEXT_SIZE;
            } else if (savedSize > MAX_TEXT_SIZE) {
                savedSize = MAX_TEXT_SIZE;
            }
            
            // 设置字体大小
            currentTextSize = savedSize;
            textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, currentTextSize);
            
            Log.d(TAG, "已加载保存的字体大小: " + currentTextSize + "sp");
        }
    }
}
