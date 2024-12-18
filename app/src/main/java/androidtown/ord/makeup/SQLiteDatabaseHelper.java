package androidtown.ord.makeup;

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import java.util.ArrayList;
import java.util.List;

public class SQLiteDatabaseHelper extends SQLiteOpenHelper {

    private static final String DATABASE_NAME = "products.db";
    private static final int DATABASE_VERSION = 1;

    public SQLiteDatabaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        String CREATE_PRODUCTS_TABLE = "CREATE TABLE products (" +
                "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "name TEXT, " +
                "color TEXT, " +
                "price TEXT)";
        db.execSQL(CREATE_PRODUCTS_TABLE);

        // 더미 데이터 추가
        db.execSQL("INSERT INTO products (name, color, price) VALUES ('Lipstick Red', '#FF0000', '$20')");
        db.execSQL("INSERT INTO products (name, color, price) VALUES ('Glossy Black', '#000000', '$15')");
        db.execSQL("INSERT INTO products (name, color, price) VALUES ('Ocean Blue', '#0000FF', '$25')");
        db.execSQL("INSERT INTO products (name, color, price) VALUES ('Sunny Yellow', '#FFFF00', '$18')");
        db.execSQL("INSERT INTO products (name, color, price) VALUES ('Emerald Green', '#00FF00', '$22')");
    }


    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL("DROP TABLE IF EXISTS products");
        onCreate(db);
    }

    public List<Product> getProductsByColor(String color) {
        List<Product> products = new ArrayList<>();
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.rawQuery("SELECT * FROM products WHERE color = ?", new String[]{color});

        if (cursor.moveToFirst()) {
            do {
                products.add(new Product(
                        cursor.getString(1),
                        cursor.getString(2),
                        cursor.getString(3)
                ));
            } while (cursor.moveToNext());
        }
        cursor.close();
        return products;
    }
}
