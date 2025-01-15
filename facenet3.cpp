#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <glob.h>  

using namespace cv;
using namespace std;

// 閾值設置
double cosine_similar_thresh = 0.363;
double l2norm_similar_thresh = 1.128;
float scoreThreshold = 0.2;
float nmsThreshold = 0.5;
float scale = 1.0;

// 获取匹配的文件路径
vector<string> getFiles(const string& pattern) {
    glob_t glob_result;
    vector<string> files;

    // 执行glob匹配
    int ret = glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
    if (ret != 0) {
        cerr << "glob() failed to find files!" << endl;
        return files;
    }

    // 将匹配到的文件路径添加到文件列表中
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        files.push_back(glob_result.gl_pathv[i]);
    }

    globfree(&glob_result);  // 释放glob资源
    return files;
}

int main() {
  
    cv::Mat source_img = cv::imread("/home/kingargroo/cpp/facenet/test1.jpg");
    if (source_img.empty()) {
        std::cerr << "Error: Could not read source image!" << std::endl;
        return -1;
    }

    // 模型路徑
    std::string fd_modelPath = "/home/kingargroo/cpp/facenet/det.onnx";
    std::string fr_modelPath = "/home/kingargroo/cpp/facenet/face2.onnx";

    // 創建人臉檢測器模型
    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(fd_modelPath, "", Size(640, 640), scoreThreshold, nmsThreshold, 1);
    if (detector.empty()) {
        std::cerr << "Face detection model failed to load!" << std::endl;
        return -1;
    } else {
        std::cout << "Face detection model loaded successfully!" << std::endl;
    }

    // 創建人臉識別模型
    Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(fr_modelPath, "");
    if (faceRecognizer.empty()) {
        std::cerr << "Face recognition model failed to load!" << std::endl;
        return -1;
    } else {
        std::cout << "Face recognition model loaded successfully!" << std::endl;
    }

    // 設置檢測器的輸入大小
    detector->setInputSize(source_img.size());

    // 檢測源圖像中的人臉
    cv::Mat sourceface, source_feature, aligned_face_source;
    detector->detect(source_img, sourceface);
    if (sourceface.rows < 1) {
        std::cerr << "Cannot find a face in image1" << std::endl;
        return 1;
    }

    // 對齊和裁剪人臉
    faceRecognizer->alignCrop(source_img, sourceface.row(0), aligned_face_source);
    cv::imwrite("aligned_source.png", aligned_face_source); // 保存對齊後的圖像

    // 提取特徵
    faceRecognizer->feature(aligned_face_source, source_feature);
    source_feature = source_feature.clone();
    

    std::vector<float> recognize_scores;
    std::vector<std::string> file_paths;

    std::string data_path = "/home/kingargroo/Downloads/dataset/*.*";
    vector<string> files = getFiles(data_path);
    int i=0;
    for (const auto& img_path : files) {
        std::cout << "Processing file: " << img_path << std::endl;

        cv::Mat dest_img = cv::imread(img_path);
        if (dest_img.empty()) {
            std::cerr << "Failed to load image: " << img_path << std::endl;
            continue;
        }

        cv::Mat destface, dest_feature, aligned_face_dest;

        detector->setInputSize(dest_img.size());
        detector->detect(dest_img, destface);
        if (destface.empty()) {
            std::cerr << "No faces detected in image: " << img_path << std::endl;
            continue;
        }

        faceRecognizer->alignCrop(dest_img, destface.row(0), aligned_face_dest);
        cv::imwrite("aligned_dest_" + std::to_string(i) + ".png", aligned_face_dest); // 保存對齊後的圖像
        i += 1;

        // 提取特徵
        faceRecognizer->feature(aligned_face_dest, dest_feature);
        dest_feature = dest_feature.clone();


        // 檢查特徵向量的尺寸
        std::cout << "Source feature size: " << source_feature.size() << std::endl;
        std::cout << "Destination feature size: " << dest_feature.size() << std::endl;

        // 使用 OpenCV 的 match 函數計算距離
        float L2_score = faceRecognizer->match(source_feature, dest_feature, cv::FaceRecognizerSF::DisType::FR_NORM_L2);
        std::cout << "Match function L2 Score: " << L2_score << std::endl;
        recognize_scores.push_back(L2_score);
        file_paths.push_back( img_path);

    }

    if (recognize_scores.empty()) {
        std::cerr << "No valid comparisons found!" << std::endl;
        return -1;
    }

   
    auto min_it = std::min_element(recognize_scores.begin(), recognize_scores.end());
    int min_index = std::distance(recognize_scores.begin(), min_it);

    if (*min_it < l2norm_similar_thresh) {
        std::cout << "Best match found at: " << file_paths[min_index] << std::endl;
        std::cout << "Matching score: " << *min_it << std::endl;
    } else {
        std::cout << "No match found below the threshold." << std::endl;
    }

    return 0;
}
