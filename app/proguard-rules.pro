# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in ${sdk.dir}/tools/proguard/proguard-android.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Add any project specific keep options here:

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# If you use Volley, uncomment the following lines:
#-keep class com.android.volley.toolbox.** { *; }
#-keep class com.android.volley.RequestQueue
#-keep class com.android.volley.Request

# If you use Gson, uncomment the following lines:
#-keep class sun.misc.Unsafe { *; }
#-keep class com.google.gson.stream.** { *; }

# If you use Retrofit, uncomment the following lines:
#-keep class retrofit2.** { *; }
#-keep interface retrofit2.** { *; }
#-dontwarn retrofit2.**

# If you use OkHttp, uncomment the following lines:
#-keep class okhttp3.** { *; }
#-keep interface okhttp3.** { *; }
#-dontwarn okhttp3.**
#-dontwarn okio.**

# Keep application classes
-keep public class * extends android.app.Activity
-keep public class * extends android.app.Application
-keep public class * extends android.app.Service
-keep public class * extends android.content.BroadcastReceiver
-keep public class * extends android.content.ContentProvider
-keep public class * extends android.app.backup.BackupAgentHelper
-keep public class * extends android.preference.Preference
-keep public class com.android.vending.licensing.ILicensingService

-keepclasseswithmembernames class * {
    native <methods>;
}

-keepclasseswithmembers class * {
    public <init>(android.content.Context, android.util.AttributeSet);
}

-keepclasseswithmembers class * {
    public <init>(android.content.Context, android.util.AttributeSet, int);
}

-keepclassmembers class * extends android.view.View {
   public <init>(android.content.Context);
   public <init>(android.content.Context, android.util.AttributeSet);
   public <init>(android.content.Context, android.util.AttributeSet, int);
   public void set*(...);
}

-keepclassmembers class * implements android.os.Parcelable {
  public static final android.os.Parcelable$Creator CREATOR;
}

-keepclassmembers class **.R$* {
    public static <fields>;
}

# Keep -keepattributes Signature
-keepattributes Signature

# Keep annotations
-keepattributes *Annotation*

# Keep enum classes
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Keep specific classes and members if using reflection
# -keep public class my.package.MyClass
# -keepclassmembers class my.package.MyClass { public <fields>; public <methods>; }

# ===== 第三方库依赖处理 =====

# AndroidX Startup 相关
-keep class androidx.startup.** { *; }
-dontwarn androidx.startup.**

# AndroidX Lifecycle 相关 - 修复ProcessLifecycleInitializer找不到的问题
-keep class androidx.lifecycle.** { *; }
-dontwarn androidx.lifecycle.**

# 保持ProcessLifecycleInitializer
-keep class androidx.lifecycle.ProcessLifecycleInitializer { *; }

# AndroidX ProfileInstaller 相关 - 修复ProfileInstallerInitializer找不到的问题
-keep class androidx.profileinstaller.** { *; }
-dontwarn androidx.profileinstaller.**

# 保持ProfileInstallerInitializer
-keep class androidx.profileinstaller.ProfileInstallerInitializer { *; }

# AndroidX Emoji2 相关 - 防止EmojiCompatInitializer找不到的问题
-keep class androidx.emoji2.** { *; }
-dontwarn androidx.emoji2.**

# 保持Emoji2初始化器
-keep class androidx.emoji2.text.EmojiCompatInitializer { *; }

# OkHttp相关
-keep class okhttp3.** { *; }
-keep interface okhttp3.** { *; }
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.codehaus.mojo.animal_sniffer.**

# Apache POI相关 - 忽略AWT和图像处理相关的可选依赖
-dontwarn java.awt.**
-dontwarn javax.imageio.**
-dontwarn org.apache.batik.**
-dontwarn com.caverock.androidsvg.**

# Saxon XML处理器 - 可选依赖
-dontwarn net.sf.saxon.**

# 其他可选依赖
-dontwarn aQute.bnd.annotation.spi.**
-dontwarn reactor.blockhound.**
-dontwarn org.codehaus.stax2.**

# javax.annotation相关 - JSR 305注解
-dontwarn javax.annotation.**
-dontwarn javax.annotation.concurrent.**

# OSGi框架相关 - 可选依赖
-dontwarn org.osgi.framework.**

# Retrofit2相关规则已移除 - 项目不再使用 Retrofit

# Apache HTTP Client相关 - Android已弃用
-dontwarn org.apache.http.**
-dontwarn android.net.http.**

# Log4j相关 - 可选依赖
-dontwarn org.apache.logging.log4j.**

# OpenNLP相关 - 可选依赖
-dontwarn opennlp.tools.**

# ONNX Runtime相关
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
-dontwarn com.microsoft.onnxruntime.**

# Markwon相关
-keep class io.noties.markwon.** { *; }
-dontwarn io.noties.markwon.**

# 保持本地LLM相关类
-keep class com.example.starlocalrag.api.** { *; }
-keep class com.example.starlocalrag.LocalLlmHandler { *; }
-keep class com.example.starlocalrag.LocalLlmHandler$** { *; }

# 保持所有应用的Fragment和Activity类
-keep class com.example.starlocalrag.*Fragment { *; }
-keep class com.example.starlocalrag.*Activity { *; }
-keep class com.example.starlocalrag.MainActivity { *; }

# 保持所有点击事件处理方法
-keepclassmembers class * {
    public void onClick(android.view.View);
    public void handle*Click*();
    public void on*Click*();
}

# 保持所有Adapter类
-keep class com.example.starlocalrag.*Adapter { *; }
-keep class com.example.starlocalrag.*Adapter$** { *; }

# 保持所有Manager和Handler类
-keep class com.example.starlocalrag.*Manager { *; }
-keep class com.example.starlocalrag.*Handler { *; }
-keep class com.example.starlocalrag.*Manager$** { *; }
-keep class com.example.starlocalrag.*Handler$** { *; }

# 保持所有应用类的公共方法和字段
-keep class com.example.starlocalrag.** {
    public <fields>;
    public <methods>;
}

# 保持View Binding相关类
-keep class com.example.starlocalrag.databinding.** { *; }

# 保持所有内部类和匿名类
-keep class com.example.starlocalrag.**$* { *; }

# 保持JNI相关
-keepclasseswithmembernames class * {
    native <methods>;
}

# 保持序列化相关
-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    private static final java.io.ObjectStreamField[] serialPersistentFields;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}
