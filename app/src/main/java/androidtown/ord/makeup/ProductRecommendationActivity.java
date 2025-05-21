package androidtown.ord.makeup;

import android.os.Bundle;
import android.widget.ListView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;

public class ProductRecommendationActivity extends AppCompatActivity {

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_product_recommendation);

        ListView listView = findViewById(R.id.productListView);
        SQLiteDatabaseHelper dbHelper = new SQLiteDatabaseHelper(this);

        // 추출된 색상 데이터 가져오기
        ArrayList<String> extractedColors = getIntent().getStringArrayListExtra("extractedColors");

        if (extractedColors != null && !extractedColors.isEmpty()) {
            List<Product> recommendedProducts = new ArrayList<>();
            for (String color : extractedColors) {
                recommendedProducts.addAll(dbHelper.getProductsByColor(color));
            }

            ProductAdapter adapter = new ProductAdapter(this, recommendedProducts);
            listView.setAdapter(adapter);
        } else {
            showToast("No colors received. Please extract colors first.");
        }
    }

    private void showToast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }

}
