package com.example.starlocalrag;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentTransaction;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.bottomnavigation.BottomNavigationView;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;
import android.provider.Settings;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;

public class MainActivity extends AppCompatActivity implements SettingsFragment.SettingsChangeListener {

    private static final String TAG = "StarLocalRAG"; // 添加TAG用于日志打印
    // 使用BuildConfig中的构建时间作为版本号，而不是实时时间
    private static final String BUILD_VERSION = BuildConfig.BUILD_VERSION;
    
    // 权限请求相关常量
    private static final int PERMISSION_REQUEST_CODE = 1001;
    private static final int MANAGE_STORAGE_REQUEST_CODE = 1002;
    private static final int BATTERY_OPTIMIZATION_REQUEST_CODE = 1003;

    private ViewPager2 viewPager;
    private BottomNavigationView bottomNavigation;
    private LogManager logManager;
    
    private boolean isInForeground = false;
    private KnowledgeBaseBuilderService knowledgeBaseBuilderService;
    private ServiceConnection serviceConnection;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // 初始化GPU错误处理
        GPUErrorHandler.init(getApplicationContext(), getWindow());
        
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_tabbed);
        
        // 初始化日志管理器
        logManager = LogManager.getInstance(this);
        
        // 加载日志配置
        LogManager.loadLogConfig(this);
        
        // 如果是Release版本，默认强制记录知识库构建过程的日志
        if (!BuildConfig.DEBUG) {
            LogManager.setForceLogToFile(true);
            LogManager.saveLogConfig(this);
        }
        
        logManager.i(TAG, "应用启动，版本: " + BUILD_VERSION);
        
        // 请求必要的权限
        requestRequiredPermissions();
        
        // 请求忽略电池优化
        requestIgnoreBatteryOptimization();
        
        // 初始化配置
        initializeConfig();
        
        // 初始化ViewPager2和BottomNavigationView
        viewPager = findViewById(R.id.viewPager);
        bottomNavigation = findViewById(R.id.bottomNavigation);
        
        // 设置ViewPager2适配器
        viewPager.setAdapter(new FragmentStateAdapter(this) {
            @NonNull
            @Override
            public Fragment createFragment(int position) {
                // 根据位置返回对应的Fragment
                switch (position) {
                    case 0:
                        return new RagQaFragment();
                    case 1:
                        return new BuildKnowledgeBaseFragment();
                    case 2:
                        return new KnowledgeNoteFragment();
                    default:
                        return new RagQaFragment();
                }
            }
            
            @Override
            public int getItemCount() {
                return 3; // 总共有3个页面
            }
        });
        
        // 禁用ViewPager2的滑动功能，只通过底部导航栏切换
        viewPager.setUserInputEnabled(false);
        
        // 设置底部导航栏的选择监听器
        bottomNavigation.setOnItemSelectedListener(item -> {
            int itemId = item.getItemId();
            if (itemId == R.id.navigation_rag_qa) {
                viewPager.setCurrentItem(0, false);
                return true;
            } else if (itemId == R.id.navigation_build_kb) {
                viewPager.setCurrentItem(1, false);
                return true;
            } else if (itemId == R.id.navigation_kb_note) {
                viewPager.setCurrentItem(2, false);
                return true;
            }
            return false;
        });
        
        // 设置ViewPager2的页面切换监听器
        viewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageSelected(int position) {
                switch (position) {
                    case 0:
                        bottomNavigation.setSelectedItemId(R.id.navigation_rag_qa);
                        break;
                    case 1:
                        bottomNavigation.setSelectedItemId(R.id.navigation_build_kb);
                        break;
                    case 2:
                        bottomNavigation.setSelectedItemId(R.id.navigation_kb_note);
                        break;
                }
            }
        });
        
        // 绑定到知识库构建服务
        bindToKnowledgeBaseBuilderService();
    }
    
    /**
     * 请求必要的权限
     */
    private void requestRequiredPermissions() {
        // 检查是否已经获得了所有权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            
            // 请求存储权限
            ActivityCompat.requestPermissions(this,
                    new String[]{
                            Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE
                    },
                    PERMISSION_REQUEST_CODE);
        }
        
        // 对于Android 11及以上版本，需要请求MANAGE_EXTERNAL_STORAGE权限
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // 检查是否已经保存了权限状态
            boolean hasStoragePermission = ConfigManager.getBoolean(this, "has_storage_permission", false);
            
            if (!hasStoragePermission && !Environment.isExternalStorageManager()) {
                try {
                    // 显示一次性权限请求对话框
                    androidx.appcompat.app.AlertDialog.Builder builder = new androidx.appcompat.app.AlertDialog.Builder(this);
                    builder.setTitle("需要完全文件访问权限");
                    builder.setMessage("StarLocalRAG需要完全的文件访问权限才能正常工作。请在接下来的设置页面中授予权限，授予后将不再请求此权限。");
                    builder.setPositiveButton("前往设置", (dialog, which) -> {
                        Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                        Uri uri = Uri.fromParts("package", getPackageName(), null);
                        intent.setData(uri);
                        startActivityForResult(intent, MANAGE_STORAGE_REQUEST_CODE);
                    });
                    builder.setNegativeButton("取消", (dialog, which) -> {
                        Toast.makeText(this, "应用可能无法正常工作", Toast.LENGTH_LONG).show();
                    });
                    builder.setCancelable(false);
                    builder.show();
                } catch (Exception e) {
                    Log.e(TAG, "无法打开文件访问权限设置: " + e.getMessage());
                    Toast.makeText(this, "无法打开文件访问权限设置，请手动授予权限", Toast.LENGTH_LONG).show();
                }
            } else if (Environment.isExternalStorageManager() && !hasStoragePermission) {
                // 如果已经有权限但没有保存状态，则保存状态
                ConfigManager.setBoolean(this, "has_storage_permission", true);
                Log.d(TAG, "已获得完全文件访问权限并保存状态");
            }
        }
    }
    
    /**
     * 请求忽略电池优化
     */
    private void requestIgnoreBatteryOptimization() {
        // 不再在应用启动时自动请求，而是在需要时才请求
        Log.d(TAG, "电池优化状态: " + (isIgnoringBatteryOptimizations() ? "已忽略" : "未忽略"));
    }
    
    /**
     * 请求忽略电池优化
     * @return 是否成功发起请求
     */
    public boolean requestIgnoreBatteryOptimizationIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!isIgnoringBatteryOptimizations()) {
                Intent intent = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
                intent.setData(Uri.parse("package:" + getPackageName()));
                try {
                    startActivityForResult(intent, BATTERY_OPTIMIZATION_REQUEST_CODE);
                    return true;
                } catch (Exception e) {
                    Log.e(TAG, "请求忽略电池优化失败: " + e.getMessage(), e);
                    return false;
                }
            }
        }
        return false;
    }
    
    /**
     * 恢复电池优化
     * @return 是否成功发起请求
     */
    public boolean restoreBatteryOptimization() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (isIgnoringBatteryOptimizations()) {
                Intent intent = new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS);
                try {
                    Toast.makeText(this, "请在设置中重新启用对本应用的电池优化", Toast.LENGTH_LONG).show();
                    startActivity(intent);
                    return true;
                } catch (Exception e) {
                    Log.e(TAG, "恢复电池优化失败: " + e.getMessage(), e);
                    return false;
                }
            }
        }
        return false;
    }
    
    /**
     * 初始化应用配置
     */
    private void initializeConfig() {
        try {
            // 加载配置，如果不存在会创建默认配置
            JSONObject config = ConfigManager.loadConfig(this);
            Log.d(TAG, "应用启动时加载配置: " + config.toString(2));
            
            // 确保配置文件存在
            File configFile = new File(getFilesDir(), ".config");
            if (configFile.exists()) {
                Log.d(TAG, "配置文件已存在: " + configFile.getAbsolutePath());
            } else {
                Log.d(TAG, "配置文件不存在，创建默认配置");
                ConfigManager.saveConfig(this, config);
            }
            
            // 初始化默认API Keys
            ConfigManager.initializeDefaultApiKeys(this);
            
            // 设置默认分块大小和重叠大小
            if (!config.has(ConfigManager.KEY_CHUNK_SIZE)) {
                ConfigManager.setInt(this, ConfigManager.KEY_CHUNK_SIZE, 1000);
            }
            if (!config.has(ConfigManager.KEY_OVERLAP_SIZE)) {
                ConfigManager.setInt(this, ConfigManager.KEY_OVERLAP_SIZE, 200);
            }
        } catch (Exception e) {
            Log.e(TAG, "初始化配置失败", e);
        }
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            
            if (allGranted) {
                Log.d(TAG, "所有权限已授予");
                // 重新加载配置
                initializeConfig();
            } else {
                Log.e(TAG, "权限被拒绝");
                Toast.makeText(this, "需要存储权限才能访问模型和知识库文件", Toast.LENGTH_LONG).show();
                
                // 显示权限说明对话框
                showPermissionExplanationDialog();
            }
        }
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        
        if (requestCode == MANAGE_STORAGE_REQUEST_CODE) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                if (Environment.isExternalStorageManager()) {
                    // 已获得权限，保存状态
                    ConfigManager.setBoolean(this, "has_storage_permission", true);
                    Log.d(TAG, "已获得完全文件访问权限");
                    Toast.makeText(this, "已获得文件访问权限", Toast.LENGTH_SHORT).show();
                } else {
                    // 未获得权限
                    Log.w(TAG, "未获得完全文件访问权限");
                    Toast.makeText(this, "未获得文件访问权限，应用可能无法正常工作", Toast.LENGTH_LONG).show();
                }
            }
        } else if (requestCode == BATTERY_OPTIMIZATION_REQUEST_CODE) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (isIgnoringBatteryOptimizations()) {
                    Log.d(TAG, "已忽略电池优化");
                } else {
                    Log.w(TAG, "未忽略电池优化");
                }
            }
        }
    }
    
    /**
     * 显示权限说明对话框
     */
    private void showPermissionExplanationDialog() {
        androidx.appcompat.app.AlertDialog.Builder builder = new androidx.appcompat.app.AlertDialog.Builder(this);
        builder.setTitle("需要存储权限");
        builder.setMessage("StarLocalRAG 需要访问外部存储权限才能读取和保存模型文件、知识库文件等。\n\n" +
                "请在设置中授予应用存储权限，否则应用将无法正常工作。");
        builder.setPositiveButton("前往设置", (dialog, which) -> {
            // 打开应用设置页面
            Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
            Uri uri = Uri.fromParts("package", getPackageName(), null);
            intent.setData(uri);
            startActivity(intent);
        });
        builder.setNegativeButton("取消", (dialog, which) -> {
            Toast.makeText(this, "应用可能无法正常工作", Toast.LENGTH_SHORT).show();
        });
        builder.setCancelable(false);
        builder.show();
    }
    
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu, menu);
        return true;
    }
    
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        
        if (id == R.id.action_settings) {
            // 打开设置界面
            viewPager.setVisibility(View.GONE);
            findViewById(R.id.container).setVisibility(View.VISIBLE);
            
            FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
            transaction.replace(R.id.container, new SettingsFragment());
            transaction.addToBackStack("settings");
            transaction.commit();
            return true;
        } else if (id == R.id.action_exit) {
            // 退出应用
            finish();
            return true;
        } else if (id == R.id.action_about) {
            // 显示关于信息
            showAboutDialog();
            return true;
        } else if (id == R.id.action_help) {
            // 打开帮助界面
            viewPager.setVisibility(View.GONE);
            findViewById(R.id.container).setVisibility(View.VISIBLE);
            
            FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
            transaction.replace(R.id.container, new HelpFragment());
            transaction.addToBackStack("help");
            transaction.commit();
            return true;
        } else if (id == R.id.action_view_log) {
            // 打开日志查看界面
            viewPager.setVisibility(View.GONE);
            findViewById(R.id.container).setVisibility(View.VISIBLE);
            
            FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
            transaction.replace(R.id.container, new LogViewFragment());
            transaction.addToBackStack("log_view");
            transaction.commit();
            return true;
        }
        
        return super.onOptionsItemSelected(item);
    }
    
    @Override
    public void onBackPressed() {
        if (findViewById(R.id.container).getVisibility() == View.VISIBLE) {
            // 如果设置页面可见，返回时恢复ViewPager2
            viewPager.setVisibility(View.VISIBLE);
            findViewById(R.id.container).setVisibility(View.GONE);
            getSupportFragmentManager().popBackStack();
        } else {
            super.onBackPressed();
        }
    }
    
    private void showAboutDialog() {
        // 使用BuildConfig中的构建时间作为版本号
        String versionInfo = BUILD_VERSION;
        
        androidx.appcompat.app.AlertDialog.Builder builder = new androidx.appcompat.app.AlertDialog.Builder(this);
        builder.setTitle("关于 StarLocalRAG");
        builder.setMessage("版本: " + versionInfo + "\n\n本地RAG (检索增强生成) 应用");
        builder.setPositiveButton("确定", null);
        builder.show();
    }
    
    @Override
    public void onSettingsChanged() {
        // 设置已更改，刷新相关数据
        Log.d(TAG, "设置已更改，刷新数据");
        
        // 重新加载当前Fragment
        int currentItem = viewPager.getCurrentItem();
        viewPager.setAdapter(viewPager.getAdapter());
        viewPager.setCurrentItem(currentItem);
    }
    
    /**
     * 导航到知识库笔记页面
     */
    public void navigateToKnowledgeNote() {
        // 切换到知识库笔记页面（索引为2）
        viewPager.setCurrentItem(2, false);
        bottomNavigation.setSelectedItemId(R.id.navigation_kb_note);
    }
    
    private boolean isIgnoringBatteryOptimizations() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
            return powerManager.isIgnoringBatteryOptimizations(getPackageName());
        }
        return false;
    }
    
    @Override
    protected void onStart() {
        super.onStart();
        Log.d(TAG, "MainActivity.onStart()");
        isInForeground = true;
        
        // 如果服务正在运行，通知服务应用已切换到前台
        if (knowledgeBaseBuilderService != null) {
            knowledgeBaseBuilderService.onAppForegrounded();
            Log.d(TAG, "已通知知识库构建服务：应用切换到前台");
        }
    }
    
    @Override
    protected void onStop() {
        super.onStop();
        Log.d(TAG, "MainActivity.onStop()");
        isInForeground = false;
        
        // 如果服务正在运行，通知服务应用已切换到后台
        if (knowledgeBaseBuilderService != null) {
            knowledgeBaseBuilderService.onAppBackgrounded();
            Log.d(TAG, "已通知知识库构建服务：应用切换到后台");
        }
    }
    
    /**
     * 绑定到知识库构建服务
     */
    private void bindToKnowledgeBaseBuilderService() {
        Log.d(TAG, "尝试绑定到知识库构建服务");
        if (serviceConnection == null) {
            serviceConnection = new ServiceConnection() {
                @Override
                public void onServiceConnected(ComponentName name, IBinder service) {
                    KnowledgeBaseBuilderService.LocalBinder binder = (KnowledgeBaseBuilderService.LocalBinder) service;
                    knowledgeBaseBuilderService = binder.getService();
                    Log.d(TAG, "已成功绑定到知识库构建服务");
                    
                    // 如果应用当前在后台，通知服务
                    if (!isInForeground && knowledgeBaseBuilderService != null) {
                        knowledgeBaseBuilderService.onAppBackgrounded();
                        Log.d(TAG, "绑定后通知服务：应用在后台");
                    }
                }
                
                @Override
                public void onServiceDisconnected(ComponentName name) {
                    Log.d(TAG, "与知识库构建服务的连接已断开");
                    knowledgeBaseBuilderService = null;
                }
            };
        }
        
        Intent intent = new Intent(this, KnowledgeBaseBuilderService.class);
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE);
    }
    
    /**
     * 解绑知识库构建服务
     */
    private void unbindKnowledgeBaseBuilderService() {
        if (serviceConnection != null) {
            try {
                unbindService(serviceConnection);
                Log.d(TAG, "已解绑知识库构建服务");
            } catch (IllegalArgumentException e) {
                Log.e(TAG, "解绑服务失败：" + e.getMessage());
            }
            knowledgeBaseBuilderService = null;
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "MainActivity.onDestroy()");
        
        // 解绑知识库构建服务
        unbindKnowledgeBaseBuilderService();
    }
}