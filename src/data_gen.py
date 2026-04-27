import pandas as pd
from faker import Faker
import random
import os
from tqdm import tqdm

def generate_products(num_products=100000, output_path="data/products.parquet"):
    fake = Faker()
    Faker.seed(42)
    random.seed(42)

    categories = [
        "Electronics", "Clothing", "Home & Kitchen", "Beauty", 
        "Sports & Outdoors", "Books", "Toys & Games", "Automotive"
    ]

    print(f"Generating {num_products} products...")
    
    products = []
    for i in tqdm(range(num_products)):
        product_id = f"PROD_{i:06d}"
        category = random.choice(categories)
        
        # More realistic titles based on category
        if category == "Electronics":
            title = f"{fake.company()} {random.choice(['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Smartwatch'])}"
        elif category == "Clothing":
            title = f"{fake.color_name()} {random.choice(['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Sneakers'])}"
        else:
            title = f"{fake.catch_phrase()}"

        description = fake.paragraph(nb_sentences=3)
        price = round(random.uniform(5.0, 2000.0), 2)
        
        # Picsum URL with a consistent seed based on index
        image_url = f"https://picsum.photos/seed/{i}/400/400"
        
        products.append({
            "product_id": product_id,
            "title": title,
            "description": description,
            "category": category,
            "price": price,
            "image_url": image_url
        })

    df = pd.DataFrame(products)
    
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    generate_products()
