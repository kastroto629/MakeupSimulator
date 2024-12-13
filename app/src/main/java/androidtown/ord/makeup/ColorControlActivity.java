package androidtown.ord.makeup;

import android.content.ContentResolver;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
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

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Objects;

public class ColorControlActivity extends AppCompatActivity {

    private String selectedFeature = ""; // lip or hair
    private int colorValue = 0;
    private int brightnessValue = 0;
    private static final String TAG = "ColorControlActivity";

    private File uploadedFile;

    private ImageView uploadedImageView;

    // ActivityResultLauncher to handle image picking result
    private final ActivityResultLauncher<Intent> imagePickerLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri selectedImage = result.getData().getData();
                    if (selectedImage != null) {
                        handleImageResult(selectedImage);
                    } else {
                        Toast.makeText(this, "Image selection failed", Toast.LENGTH_SHORT).show();
                    }
                } else {
                    Toast.makeText(this, "Image selection canceled", Toast.LENGTH_SHORT).show();
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_color_control);

        // View 초기화
        uploadedImageView = findViewById(R.id.uploadedImageView);
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
            imagePickerLauncher.launch(intent);
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
                colorValue = progress;
                Log.d(TAG, "Color SeekBar progress: " + progress);
                sendAdjustColorRequest();
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
                sendAdjustColorRequest();
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

    private void handleImageResult(Uri selectedImage) {
        uploadedImageView.setImageURI(selectedImage); // 선택된 이미지 표시
        Log.d(TAG, "Image uploaded successfully");

        // 파일 복사 및 저장
        try {
            ContentResolver resolver = getContentResolver();
            InputStream inputStream = resolver.openInputStream(selectedImage);

            if (inputStream == null) {
                throw new NullPointerException("InputStream is null");
            }

            String fileName = getFileName(selectedImage);
            File tempFile = new File(getCacheDir(), Objects.requireNonNull(fileName));
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
            Log.e(TAG, "Error processing the image", e);
            Toast.makeText(this, "Failed to process the image", Toast.LENGTH_SHORT).show();
        }
    }

    private void sendAdjustColorRequest() {
        if (uploadedFile == null) {
            Toast.makeText(this, "Please upload an image first", Toast.LENGTH_SHORT).show();
            return;
        }

        new Thread(() -> {
            Bitmap adjustedImage = ApiClient.adjustColor(selectedFeature, brightnessValue, colorValue, uploadedFile);
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
        if ("content".equals(uri.getScheme())) {
            try (Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    int nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                    if (nameIndex >= 0) {
                        result = cursor.getString(nameIndex);
                    }
                }
            }
        }
        if (result == null) {
            result = uri.getPath();
            if (result != null) {
                int cut = result.lastIndexOf('/');
                if (cut != -1) {
                    result = result.substring(cut + 1);
                }
            }
        }
        return result;
    }
}
