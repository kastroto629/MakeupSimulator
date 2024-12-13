package androidtown.ord.makeup;

import android.content.ContentResolver;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class ColorControlActivity extends AppCompatActivity {

    private String selectedFeature = ""; // "lip" or "hair"
    private int colorR = 0, colorG = 0, colorB = 0;
    private int brightnessValue = 0;
    private static final String TAG = "ColorControlActivity";

    private File uploadedFile;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_color_control);

        // View 초기화
        ImageView uploadedImageView = findViewById(R.id.uploadedImageView);
        Button btnUploadImage = findViewById(R.id.btnUploadImage);
        Button btnLipColor = findViewById(R.id.btnLipColor);
        Button btnHairColor = findViewById(R.id.btnHairColor);
        SeekBar seekBarColor = findViewById(R.id.seekBarColor);
        SeekBar seekBarBrightness = findViewById(R.id.seekBarBrightness);
        LinearLayout sliderLayout = findViewById(R.id.sliderLayout);
        Button btnHome = findViewById(R.id.btnHome);
        Button btnNext = findViewById(R.id.btnNext);

        // 초기 설정: 슬라이더 숨김
        sliderLayout.setVisibility(View.GONE);

        // 사진 업로드 버튼 동작
        btnUploadImage.setOnClickListener(view -> {
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            startActivityForResult(intent, 1); // 결과를 받아오기 위해 코드 1 사용
        });

        // Lip Color 버튼 동작
        btnLipColor.setOnClickListener(view -> {
            selectedFeature = "lip";
            sliderLayout.setVisibility(View.VISIBLE); // 슬라이더 표시
            Log.d(TAG, "Lip color slider displayed");
        });

        // Hair Color 버튼 동작
        btnHairColor.setOnClickListener(view -> {
            selectedFeature = "hair";
            sliderLayout.setVisibility(View.VISIBLE); // 슬라이더 표시
            Log.d(TAG, "Hair color slider displayed");
        });

        // 색상 조절 슬라이더
        seekBarColor.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                colorR = (progress & 0xFF0000) >> 16; // Red
                colorG = (progress & 0x00FF00) >> 8;  // Green
                colorB = (progress & 0x0000FF);       // Blue
                Log.d(TAG, "Color SeekBar progress: R=" + colorR + ", G=" + colorG + ", B=" + colorB);
                sendAdjustColorRequest(uploadedImageView);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                Log.d(TAG, "Color SeekBar touch started");
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                Log.d(TAG, "Color SeekBar touch stopped");
            }
        });

        // 밝기 조절 슬라이더
        seekBarBrightness.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                brightnessValue = progress;
                Log.d(TAG, "Brightness SeekBar progress: " + progress);
                sendAdjustColorRequest(uploadedImageView);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                Log.d(TAG, "Brightness SeekBar touch started");
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                Log.d(TAG, "Brightness SeekBar touch stopped");
            }
        });

        // 홈 버튼 동작
        btnHome.setOnClickListener(view -> {
            Log.d(TAG, "Home button clicked");
            finish(); // 현재 화면 종료
        });

        // 다음 화면 버튼 동작
        btnNext.setOnClickListener(view -> {
            Log.d(TAG, "Next button clicked");
            Intent intent = new Intent(ColorControlActivity.this, NextActivity.class);
            startActivity(intent); // 다음 화면으로 이동
        });
    }

    // 이미지 업로드 처리
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 1 && resultCode == RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            ImageView uploadedImageView = findViewById(R.id.uploadedImageView);
            uploadedImageView.setImageURI(selectedImage); // 선택된 이미지 표시
            Log.d(TAG, "Image uploaded successfully");

            // 파일 복사 및 저장
            try {
                ContentResolver resolver = getContentResolver();
                InputStream inputStream = resolver.openInputStream(selectedImage);

                String fileName = getFileName(selectedImage);
                File tempFile = new File(getCacheDir(), fileName);
                FileOutputStream outputStream = new FileOutputStream(tempFile);

                byte[] buffer = new byte[1024];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
                outputStream.close();
                inputStream.close();

                uploadedFile = tempFile; // 저장된 파일 경로 저장
                Log.d(TAG, "File saved: " + uploadedFile.getAbsolutePath());

            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "Failed to process the image", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // 색상 및 밝기 조절 API 요청
    private void sendAdjustColorRequest(ImageView uploadedImageView) {
        if (uploadedFile == null) {
            Toast.makeText(this, "Please upload an image first", Toast.LENGTH_SHORT).show();
            return;
        }

        new Thread(() -> {
            Bitmap adjustedImage = ApiClient.adjustColor(selectedFeature, brightnessValue, Color.rgb(colorR, colorG, colorB), uploadedFile);
            runOnUiThread(() -> {
                if (adjustedImage != null) {
                    uploadedImageView.setImageBitmap(adjustedImage);
                    Log.d(TAG, "Image adjusted successfully");
                } else {
                    Toast.makeText(this, "Failed to adjust color", Toast.LENGTH_SHORT).show();
                    Log.e(TAG, "Failed to adjust color");
                }
            });
        }).start();
    }

    private String getFileName(Uri uri) {
        String result = null;
        if (uri.getScheme().equals("content")) {
            try (Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    result = cursor.getString(cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME));
                }
            }
        }
        if (result == null) {
            result = uri.getPath();
            int cut = result.lastIndexOf('/');
            if (cut != -1) {
                result = result.substring(cut + 1);
            }
        }
        return result;
    }
}
