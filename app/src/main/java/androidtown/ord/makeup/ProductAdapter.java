package androidtown.ord.makeup;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

import java.util.List;

public class ProductAdapter extends BaseAdapter {
    private Context context;
    private List<Product> productList;

    public ProductAdapter(Context context, List<Product> productList) {
        this.context = context;
        this.productList = productList;
    }

    @Override
    public int getCount() {
        return productList.size();
    }

    @Override
    public Object getItem(int position) {
        return productList.get(position);
    }

    @Override
    public long getItemId(int position) {
        return position;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        if (convertView == null) {
            convertView = LayoutInflater.from(context).inflate(R.layout.item_product, parent, false);
        }

        TextView nameTextView = convertView.findViewById(R.id.productName);
        TextView colorTextView = convertView.findViewById(R.id.productColor);
        TextView priceTextView = convertView.findViewById(R.id.productPrice);

        Product product = productList.get(position);
        nameTextView.setText(product.getName());
        colorTextView.setText(product.getColor());
        priceTextView.setText(product.getPrice());

        return convertView;
    }
}
