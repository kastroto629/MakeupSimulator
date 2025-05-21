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
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ColorExtractActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    private ImageView imageView;
    private LinearLayout colorResultLayout;
    private Bitmap selectedImageBitmap;
    private List<String> extractedColors; // k-means로 추출된 색상 저장

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_color_extract);

        imageView = findViewById(R.id.imageView);
        Button uploadImageButton = findViewById(R.id.uploadImageButton);
        Button homeButton = findViewById(R.id.homeButton);
        Button recommendButton = findViewById(R.id.recommendButton);
        colorResultLayout = findViewById(R.id.colorResultLayout);
        extractedColors = new ArrayList<>();

        // 이미지 업로드 버튼 클릭
        uploadImageButton.setOnClickListener(v -> openImageChooser());

        // 홈 버튼 클릭
        homeButton.setOnClickListener(v -> {
            Intent intent = new Intent(ColorExtractActivity.this, MainActivity.class);
            startActivity(intent);
            finish();
        });

        // 추천 버튼 클릭
        recommendButton.setOnClickListener(v -> {
            if (!extractedColors.isEmpty()) {
                Intent intent = new Intent(ColorExtractActivity.this, ProductRecommendationActivity.class);
                intent.putStringArrayListExtra("extractedColors", new ArrayList<>(extractedColors));
                startActivity(intent);
            } else {
                showToast("No colors extracted. Please upload an image first.");
            }
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
                extractColorsUsingKMeans(selectedImageBitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void extractColorsUsingKMeans(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        List<int[]> pixels = new ArrayList<>();

        // 이미지의 모든 픽셀 RGB 값 수집
        for (int y = 0; y < height; y += 10) {  // 샘플링을 위해 10픽셀 단위
            for (int x = 0; x < width; x += 10) {
                int pixel = bitmap.getPixel(x, y);
                int r = Color.red(pixel);
                int g = Color.green(pixel);
                int b = Color.blue(pixel);
                pixels.add(new int[]{r, g, b});
            }
        }

        // k-means 실행 (k=10)
        List<int[]> centroids = kMeans(pixels, 10);

        // 추출된 색상 추가 및 UI에 표시
        extractedColors.clear();
        colorResultLayout.removeAllViews();

        for (int[] centroid : centroids) {
            String hexColor = String.format("#%02X%02X%02X", centroid[0], centroid[1], centroid[2]);
            extractedColors.add(hexColor);
            addColorToResultLayout(hexColor);
        }
    }

    private List<int[]> kMeans(List<int[]> pixels, int k) {
        List<int[]> centroids = new ArrayList<>();
        // 초기 랜덤 중심점 설정
        for (int i = 0; i < k; i++) {
            centroids.add(pixels.get(i * pixels.size() / k));
        }

        boolean centroidsChanged;
        do {
            centroidsChanged = false;
            List<List<int[]>> clusters = new ArrayList<>();
            for (int i = 0; i < k; i++) clusters.add(new ArrayList<>());

            // 각 픽셀을 가장 가까운 클러스터에 할당
            for (int[] pixel : pixels) {
                int closestIndex = 0;
                double minDistance = Double.MAX_VALUE;
                for (int i = 0; i < k; i++) {
                    double distance = euclideanDistance(pixel, centroids.get(i));
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestIndex = i;
                    }
                }
                clusters.get(closestIndex).add(pixel);
            }

            // 새로운 중심점 계산
            for (int i = 0; i < k; i++) {
                if (!clusters.get(i).isEmpty()) {
                    int[] newCentroid = calculateMean(clusters.get(i));
                    if (!isEqual(centroids.get(i), newCentroid)) {
                        centroidsChanged = true;
                        centroids.set(i, newCentroid);
                    }
                }
            }
        } while (centroidsChanged);

        return centroids;
    }

    private double euclideanDistance(int[] a, int[] b) {
        return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2) + Math.pow(a[2] - b[2], 2));
    }

    private int[] calculateMean(List<int[]> cluster) {
        int sumR = 0, sumG = 0, sumB = 0;
        for (int[] pixel : cluster) {
            sumR += pixel[0];
            sumG += pixel[1];
            sumB += pixel[2];
        }
        int size = cluster.size();
        return new int[]{sumR / size, sumG / size, sumB / size};
    }

    private boolean isEqual(int[] a, int[] b) {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }

    private void addColorToResultLayout(String hexColor) {
        LinearLayout colorItemLayout = new LinearLayout(this);
        colorItemLayout.setOrientation(LinearLayout.HORIZONTAL);

        View colorView = new View(this);
        colorView.setLayoutParams(new LinearLayout.LayoutParams(100, 100));
        colorView.setBackgroundColor(Color.parseColor(hexColor));

        Button colorCodeButton = new Button(this);
        colorCodeButton.setText(hexColor);

        colorItemLayout.addView(colorView);
        colorItemLayout.addView(colorCodeButton);
        colorResultLayout.addView(colorItemLayout);
    }

    private void showToast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }
}
