package androidtown.ord.makeup;

public class Product {
    private final String name;
    private final String color;
    private final String price;

    public Product(String name, String color, String price) {
        this.name = name;
        this.color = color;
        this.price = price;
    }

    public String getName() {
        return name;
    }

    public String getColor() {
        return color;
    }

    public String getPrice() {
        return price;
    }
}
