package androidtown.ord.makeup;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.logging.HttpLoggingInterceptor;

public class ApiClient {

    private static final String SERVER_URL = "http://10.0.2.2:8000/adjust-color/"; // Emulator-specific address
    private static final MediaType MEDIA_TYPE_JPEG = MediaType.parse("image/jpeg");

    public static Bitmap adjustColor(String feature, int brightness, int color, File imageFile) {
        // Logging Interceptor
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);

        OkHttpClient client = new OkHttpClient.Builder()
                .addInterceptor(logging)
                .connectTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .build();

        // Convert image file to MultipartBody
        RequestBody imageRequestBody = RequestBody.create(imageFile, MEDIA_TYPE_JPEG);
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("feature", feature)
                .addFormDataPart("brightness", String.valueOf(brightness))
                .addFormDataPart("color_r", String.valueOf((color >> 16) & 0xFF))
                .addFormDataPart("color_g", String.valueOf((color >> 8) & 0xFF))
                .addFormDataPart("color_b", String.valueOf(color & 0xFF))
                .addFormDataPart("file", "image.jpg", imageRequestBody)
                .build();

        Request request = new Request.Builder()
                .url(SERVER_URL)
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                // Decode response image
                byte[] imageBytes = response.body().bytes();
                return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            } else {
                Log.e("ApiClient", "Request failed: " + response.message());
            }
        } catch (IOException e) {
            Log.e("ApiClient", "Network error occurred: ", e);
        }
        return null;
    }
}
