package androidtown.ord.makeup;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

public class ColorExtractActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    private ImageView imageView;
    private LinearLayout colorResultLayout;
    private Bitmap selectedImageBitmap;
    private Set<String> uniqueColors;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_color_extract);

        imageView = findViewById(R.id.imageView);
        Button uploadImageButton = findViewById(R.id.uploadImageButton);
        Button homeButton = findViewById(R.id.homeButton);
        colorResultLayout = findViewById(R.id.colorResultLayout);
        uniqueColors = new HashSet<>();

        // 이미지 업로드 버튼 클릭
        uploadImageButton.setOnClickListener(v -> openImageChooser());

        // 홈 버튼 클릭
        homeButton.setOnClickListener(v -> {
            Intent intent = new Intent(ColorExtractActivity.this, MainActivity.class);
            startActivity(intent);
            finish();
        });
    }

    // 이미지 선택 함수
    private void openImageChooser() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            Uri imageUri = data.getData();
            try {
                selectedImageBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                imageView.setImageBitmap(selectedImageBitmap);
                extractColorsFromImage(selectedImageBitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // 색상 추출 함수
    private void extractColorsFromImage(Bitmap bitmap) {
        uniqueColors.clear();
        colorResultLayout.removeAllViews();

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // 이미지의 각 픽셀 색상 가져오기
        for (int y = 0; y < height; y += 10) { // 10px 단위로 건너뛰어 성능 최적화
            for (int x = 0; x < width; x += 10) {
                int pixel = bitmap.getPixel(x, y);
                String hexColor = String.format("#%06X", (0xFFFFFF & pixel));

                if (uniqueColors.add(hexColor)) {
                    addColorToResultLayout(hexColor);
                }
            }
        }
    }

    // 색상 결과를 동적으로 레이아웃에 추가
    private void addColorToResultLayout(String hexColor) {
        LinearLayout colorItemLayout = new LinearLayout(this);
        colorItemLayout.setOrientation(LinearLayout.HORIZONTAL);

        // 색상 이미지
        View colorView = new View(this);
        colorView.setLayoutParams(new LinearLayout.LayoutParams(100, 100));
        colorView.setBackgroundColor(Color.parseColor(hexColor));

        // 색상 코드 버튼
        Button colorCodeButton = new Button(this);
        colorCodeButton.setText(hexColor);
        colorCodeButton.setOnClickListener(v -> {
            // 버튼 클릭 시 동작 추가 (필요에 따라)
        });

        // 색상 이미지와 코드 버튼 추가
        colorItemLayout.addView(colorView);
        colorItemLayout.addView(colorCodeButton);
        colorResultLayout.addView(colorItemLayout);
    }
}
