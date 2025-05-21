package androidtown.ord.makeup;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.json.JSONObject;

import java.io.File;
import java.io.IOException;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class ApiClient {

    private static final String TAG = "ApiClient";
    private static final String BASE_URL = "http://10.0.2.2:8000";
    private static final MediaType MEDIA_TYPE_JPEG = MediaType.parse("image/jpeg");
    private static final OkHttpClient client = new OkHttpClient();

    public static String uploadImage(File imageFile) {
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", imageFile.getName(),
                        RequestBody.create(imageFile, MEDIA_TYPE_JPEG))
                .build();

        Request request = new Request.Builder()
                .url(BASE_URL + "/upload-image/")
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful() && response.body() != null) {
                JSONObject jsonResponse = new JSONObject(response.body().string());
                return jsonResponse.getString("session_id");
            } else {
                Log.e(TAG, "Error: " + response.code());
            }
        } catch (Exception e) {
            Log.e(TAG, "Exception uploading image", e);
        }
        return null;
    }

    public static Bitmap adjustColor(String sessionId, String feature, int brightness, int r, int g, int b) {
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("session_id", sessionId)
                .addFormDataPart("feature", feature)
                .addFormDataPart("brightness", String.valueOf(brightness))
                .addFormDataPart("color_r", String.valueOf(r))
                .addFormDataPart("color_g", String.valueOf(g))
                .addFormDataPart("color_b", String.valueOf(b))
                .build();

        Request request = new Request.Builder()
                .url(BASE_URL + "/adjust-color/")
                .post(requestBody)
                .build();

        return fetchImageFromResponse(request);
    }

    public static Bitmap resetImage(String sessionId) {
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("session_id", sessionId)
                .build();

        Request request = new Request.Builder()
                .url(BASE_URL + "/reset/")
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful() && response.body() != null) {
                byte[] imageBytes = response.body().bytes();
                return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            }
        } catch (IOException e) {
            Log.e(TAG, "Error resetting image", e);
        }
        return null;
    }


    private static Bitmap fetchImageFromResponse(Request request) {
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful() && response.body() != null) {
                byte[] imageBytes = response.body().bytes();
                return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            } else {
                Log.e(TAG, "Error fetching image: " + response.code());
            }
        } catch (IOException e) {
            Log.e(TAG, "Exception in image fetch", e);
        }
        return null;
    }
}
