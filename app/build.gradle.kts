plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("kotlin-kapt") // KAPT 플러그인 추가
}

android {
    namespace = "androidtown.ord.makeup"
    compileSdk = 33

    defaultConfig {
        applicationId = "androidtown.ord.makeup"
        minSdk = 21
        targetSdk = 33
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    buildFeatures {
        viewBinding = true
    }

    lint {
        abortOnError = false
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }
}

dependencies {
    // AndroidX Libraries
    implementation("androidx.core:core-ktx:1.10.1")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.9.0")

    // Activity and Lifecycle Components
    implementation("androidx.activity:activity:1.6.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.1")

    // Networking (OkHttp)
    implementation("com.squareup.okhttp3:okhttp:4.11.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.11.0")
    implementation ("androidx.recyclerview:recyclerview:1.3.1")

    // Image Processing (Optional)
    implementation("com.github.bumptech.glide:glide:4.15.1")
    kapt("com.github.bumptech.glide:compiler:4.15.1") // Glide Compiler for Annotation Processing
}
