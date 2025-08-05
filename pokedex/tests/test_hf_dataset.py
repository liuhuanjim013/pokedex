# Copy content from scripts/yolo/test_hf_dataset.py
#!/usr/bin/env python3
"""
Quick test script for Hugging Face dataset access.
Copy this entire script into a Colab cell to test dataset access.
"""

def test_hf_dataset():
    """Test Hugging Face dataset access."""
    import os
    from datasets import load_dataset
    from huggingface_hub import login, HfApi

    print("🔍 Testing Hugging Face setup...")

    # 1. Test environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("⚠️ HUGGINGFACE_TOKEN not found in environment")
        print("ℹ️ Will try to access dataset without token...")
    else:
        print("✅ Found HUGGINGFACE_TOKEN")
        login(token=hf_token)

    try:
        # 2. Test public dataset access
        print("\n📊 Testing public dataset access...")
        test_dataset = load_dataset("mnist", split="train[:1]")
        print("✅ Public dataset access works")

        # 3. Test API connection
        print("\n🔑 Testing API connection...")
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Connected as: {user_info['name']} ({user_info['fullname']})")

        # 4. Test Pokemon dataset
        print("\n🐉 Testing Pokemon dataset access...")
        dataset = load_dataset("liuhuanjim013/pokemon-yolo-1025")
        print("✅ Pokemon dataset loaded!")
        print(f"\nDataset info:")
        print(f"• Splits: {list(dataset.keys())}")
        print(f"• Training examples: {len(dataset['train'])}")
        print(f"• Features: {list(dataset['train'].features.keys())}")

        # 5. Test loading a single example
        print("\n📸 Testing example access...")
        example = dataset['train'][0]
        print("✅ Successfully loaded first example")
        print(f"• Image type: {type(example['image']).__name__}")
        print(f"• Label: {example['label'] if 'label' in example else 'No label'}")

        # 6. Test loading and converting image
        print("\n🖼️ Testing image loading...")
        from PIL import Image
        import numpy as np
        
        if isinstance(example['image'], Image.Image):
            # If image is a PIL Image
            print(f"✅ Image is a PIL Image")
            print(f"• Image mode: {example['image'].mode}")
            print(f"• Image size: {example['image'].size}")
            img_array = np.array(example['image'])
        elif isinstance(example['image'], str):
            # If image is a path
            img = Image.open(example['image'])
            img_array = np.array(img)
            print(f"✅ Successfully loaded image from path")
        else:
            # If image is already an array
            img_array = example['image']
            print(f"✅ Image is already in array format")
        
        print(f"• Array shape: {img_array.shape}")
        print(f"• Array dtype: {img_array.dtype}")
        print(f"• Value range: [{img_array.min()}, {img_array.max()}]")

        print("\n🎉 All tests passed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Verify HUGGINGFACE_TOKEN is correct")
        print("3. Try running: huggingface-cli login")
        print("4. Check dataset URL: https://huggingface.co/datasets/liuhuanjim013/pokemon-yolo-1025")
        return False

if __name__ == "__main__":
    # Install required packages if needed
    INSTALL_COMMAND = """
    !pip install --quiet datasets huggingface_hub
    """
    print("📦 If packages are missing, run this first:")
    print(INSTALL_COMMAND)
    print("\n" + "="*60 + "\n")
    
    test_hf_dataset()