package androidtown.ord.makeup;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
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
import java.io.IOException;
import java.io.InputStream;

import android.graphics.drawable.BitmapDrawable;
import android.os.Environment;
import java.io.FileOutputStream;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;


public class ColorControlActivity extends AppCompatActivity {

    private static final String TAG = "ColorControlActivity";

    private ImageView uploadedImageView;
    private String sessionId;
    private String selectedFeature = "lip";
    private int redValue = 0, greenValue = 0, blueValue = 0, brightnessValue = 0;



    private final ActivityResultLauncher<Intent> imagePickerLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri selectedImage = result.getData().getData();
                    uploadedImageView.setImageURI(selectedImage);
                    Log.d(TAG, "Image uploaded successfully.");
                    uploadImageToServer(selectedImage);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_color_control);

        // Initialize views
        uploadedImageView = findViewById(R.id.uploadedImageView);
        Button btnUploadImage = findViewById(R.id.btnUploadImage);
        Button btnLipColor = findViewById(R.id.btnLipColor);
        Button btnEyeColor = findViewById(R.id.btnEyeColor);
        Button btnReset = findViewById(R.id.btnReset);
        Button btnDown = findViewById(R.id.btnDown);
        Button btnHome = findViewById(R.id.btnHome);
        SeekBar seekBarR = findViewById(R.id.seekBarR);
        SeekBar seekBarG = findViewById(R.id.seekBarG);
        SeekBar seekBarB = findViewById(R.id.seekBarB);
        SeekBar seekBarBrightness = findViewById(R.id.seekBarBrightness);
        LinearLayout sliderLayout = findViewById(R.id.sliderLayout);


        // Initially hide the slider layout
        sliderLayout.setVisibility(View.GONE);

        // Button actions
        btnUploadImage.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            imagePickerLauncher.launch(intent);
        });

        btnLipColor.setOnClickListener(v -> {
            selectedFeature = "lip";
            sliderLayout.setVisibility(View.VISIBLE);
        });

        btnEyeColor.setOnClickListener(v -> {
            selectedFeature = "eye";
            sliderLayout.setVisibility(View.VISIBLE);
        });

        btnDown.setOnClickListener(v -> {
            // Retrieve the bitmap from ImageView
            BitmapDrawable drawable = (BitmapDrawable) uploadedImageView.getDrawable();
            if (drawable != null) {
                Bitmap bitmap = drawable.getBitmap();
                saveBitmapToGallery(bitmap);
            } else {
                Toast.makeText(this, "No image to download", Toast.LENGTH_SHORT).show();
            }
        });

        btnHome.setOnClickListener(v -> {
            // MainActivity로 이동
            Intent intent = new Intent(ColorControlActivity.this, MainActivity.class);
            // 이전 액티비티 스택을 모두 비우고 새로 시작 (선택사항)
            intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            startActivity(intent);
            finish(); // 현재 액티비티 종료
        });

        btnReset.setOnClickListener(v -> resetImage());

        seekBarR.setOnSeekBarChangeListener(createSeekBarListener(value -> redValue = value));
        seekBarG.setOnSeekBarChangeListener(createSeekBarListener(value -> greenValue = value));
        seekBarB.setOnSeekBarChangeListener(createSeekBarListener(value -> blueValue = value));
        seekBarBrightness.setOnSeekBarChangeListener(createSeekBarListener(value -> brightnessValue = value));
    }


    private SeekBar.OnSeekBarChangeListener createSeekBarListener(ValueUpdateListener listener) {
        return new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) { // 사용자가 직접 SeekBar를 조작했을 때만 처리
                    listener.onUpdate(progress);
                    sendAdjustColorRequest();
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // SeekBar 조작이 끝난 후 최종값 반영
                sendAdjustColorRequest();
            }
        };
    }


    private void resetImage() {
        if (sessionId == null) {
            Toast.makeText(this, "Please upload an image first", Toast.LENGTH_SHORT).show();
            return;
        }

        Log.d(TAG, "Reset button clicked. Sending reset request...");
        new Thread(() -> {
            Bitmap resetImage = ApiClient.resetImage(sessionId);
            runOnUiThread(() -> {
                if (resetImage != null) {
                    uploadedImageView.setImageBitmap(resetImage);

                    // SeekBar 값 초기화
                    SeekBar seekBarR = findViewById(R.id.seekBarR);
                    SeekBar seekBarG = findViewById(R.id.seekBarG);
                    SeekBar seekBarB = findViewById(R.id.seekBarB);
                    SeekBar seekBarBrightness = findViewById(R.id.seekBarBrightness);

                    seekBarR.setProgress(0);
                    seekBarG.setProgress(0);
                    seekBarB.setProgress(0);
                    seekBarBrightness.setProgress(0);

                    // 변수 값 초기화
                    redValue = 0;
                    greenValue = 0;
                    blueValue = 0;
                    brightnessValue = 0;

                    Toast.makeText(this, "Image and settings reset to original!", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(this, "Failed to reset image", Toast.LENGTH_SHORT).show();
                }
            });
        }).start();
    }


    private boolean isRequestInProgress = false; // 중복 요청 방지 플래그

    private void sendAdjustColorRequest() {
        if (sessionId == null) {
            Toast.makeText(this, "Please upload an image first", Toast.LENGTH_SHORT).show();
            return;
        }

        // 요청 중이면 실행하지 않음
        if (isRequestInProgress) return;

        isRequestInProgress = true; // 요청 시작

        new Thread(() -> {
            // 서버로부터 조정된 이미지 가져오기
            Bitmap adjustedImage = ApiClient.adjustColor(sessionId, selectedFeature, brightnessValue, redValue, greenValue, blueValue);

            runOnUiThread(() -> {
                if (adjustedImage != null) {
                    uploadedImageView.setImageBitmap(adjustedImage); // 이미지 업데이트
                } else {
                    Toast.makeText(this, "Failed to adjust color", Toast.LENGTH_SHORT).show();
                }
                isRequestInProgress = false; // 요청 완료
            });
        }).start();
    }


    private void uploadImageToServer(Uri imageUri) {
        try {
            File imageFile = createTempFileFromUri(imageUri);
            new Thread(() -> {
                sessionId = ApiClient.uploadImage(imageFile);
                runOnUiThread(() -> {
                    if (sessionId != null) {
                        Toast.makeText(this, "Image uploaded successfully!", Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(this, "Failed to upload image.", Toast.LENGTH_SHORT).show();
                    }
                });
            }).start();
        } catch (Exception e) {
            Log.e(TAG, "Error uploading image", e);
        }
    }

    private File createTempFileFromUri(Uri uri) throws IOException {
        InputStream inputStream = getContentResolver().openInputStream(uri);
        File tempFile = new File(getCacheDir(), "temp_image.jpg");
        try (FileOutputStream outputStream = new FileOutputStream(tempFile)) {
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
        }
        return tempFile;
    }


    private void saveBitmapToGallery(Bitmap bitmap) {
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = "MakeupImage_" + timestamp + ".jpg";

        // Path to external storage
        File storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        File imageFile = new File(storageDir, filename);

        try (FileOutputStream out = new FileOutputStream(imageFile)) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out); // Save bitmap as JPEG
            Toast.makeText(this, "Image saved to: " + imageFile.getAbsolutePath(), Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Failed to save image", Toast.LENGTH_SHORT).show();
        }
    }


    private interface ValueUpdateListener {
        void onUpdate(int value);
    }
}
