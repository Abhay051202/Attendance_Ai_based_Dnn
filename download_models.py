import os
import urllib.request

def download_file(url, save_path):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Saved to {save_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    data_dir = os.path.join(os.getcwd(), 'data', 'models')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # URLs for ResNet-10 SSD model (OpenCV Face Detector)
    proto_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    download_file(proto_url, os.path.join(data_dir, "deploy.prototxt"))
    download_file(model_url, os.path.join(data_dir, "res10_300x300_ssd_iter_140000.caffemodel"))

if __name__ == "__main__":
    main()
