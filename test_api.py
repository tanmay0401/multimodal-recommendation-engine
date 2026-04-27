import requests
import json

def test_recommendation(user_id=None, image_path=None):
    url = "http://127.0.0.1:8000/recommend"
    
    data = {"k": 5}
    if user_id:
        data["user_id"] = user_id
        
    files = None
    if image_path:
        files = {"image": open(image_path, "rb")}
    
    print(f"\nSending request to {url} (User: {user_id}, Image: {image_path})...")
    response = requests.post(url, data=data, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Latency: {result['retrieval_latency_ms']}ms")
        print("Top 5 Recommendations:")
        for i, res in enumerate(result['recommendations']):
            print(f"{i+1}. {res['title']} | Category: {res['category']} | Price: ${res['price']} | Score: {res['ranking_score']:.4f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Test 1: User-based
    test_recommendation(user_id="USER_00042")
    
    # Test 2: Multi-modal (User + Mock Image)
    # Since I don't have a real image handy, I'll just do another user test for now
    # or I could download one
    print("\nDownloading test image...")
    img_data = requests.get("https://picsum.photos/400/400").content
    with open("test_image.jpg", "wb") as f:
        f.write(img_data)
        
    test_recommendation(user_id="USER_00042", image_path="test_image.jpg")
