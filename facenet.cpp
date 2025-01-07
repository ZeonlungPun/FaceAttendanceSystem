#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include<onnxruntime_cxx_api.h>

void FaceDetect(std::vector<cv::Rect>& faces_coordnidates,std::string xml_path,cv::Mat& frame,bool visualize= true)
{
    cv::CascadeClassifier faceCascade;
    faceCascade.load(xml_path);
    cv::Mat grayFrame;
    cv::Mat frame_copy=frame.clone();
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    faceCascade.detectMultiScale(grayFrame, faces_coordnidates, 1.1, 3, 0, cv::Size(30, 30));

    if (visualize)
    {
        for (size_t i = 0; i < faces_coordnidates.size(); i++)
        {
            cv::rectangle(frame_copy, faces_coordnidates[i], cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite("result.png",frame_copy);
    }
    

}
Ort::Env* env = nullptr;
Ort::Session* session = nullptr;
// Lazy initialization of session options
Ort::SessionOptions& get_session_options() {
    static Ort::SessionOptions session_options;
    static bool is_initialized = false;

    if (!is_initialized) {
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        is_initialized = true;
    }

    return session_options;
}
void initialize_model(const std::string& onnx_path_name) {
    try {
        if (!env) {
            env = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "onnx");
        }

        Ort::SessionOptions& session_options = get_session_options();  // Use the lazy initialized session_options

        if (!session) {
            //std::wstring modelPath = std::wstring(onnx_path_name.begin(), onnx_path_name.end());
            session = new Ort::Session(*env, onnx_path_name.c_str(), session_options);
            std::cout << "Model session created successfully." << std::endl;
        }


    } catch (const Ort::Exception& e) {
        std::cerr << "Error during ONNX Runtime initialization: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred during initialization." << std::endl;
    }
}

cv::Mat FaceNetInference(std::string model_path, cv::Mat& frame) {
// get the FaceNet embeding vector
    // 獲取模型輸入和輸出信息
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    size_t numInputNodes = session->GetInputCount();
    size_t numOutputNodes = session->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 解析模型輸入信息
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    // 解析模型輸出信息
    int output_dim = 0;
    Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_dim = output_dims[1];

    std::cout << "output format : Nx dim = " << output_dims[0] << "x" << output_dim << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }

    // 處理輸入圖像
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    // 調整圖像大小
    if (image.cols != input_w || image.rows != input_h) {
        cv::resize(image, image, cv::Size(input_w, input_h));
    }

    // 創建 blob
    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);

    // 確認 blob 的尺寸
    std::cout << "Blob shape: " << blob.size() << std::endl;

    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3 ,input_h, input_w };
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };

    // 進行推理
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    input_tensor_.release();

    // 解析輸出結果
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat embeding_vector(1, output_dim, CV_32F, (float*)pdata);
    return embeding_vector;
}

float calculateDistance(std::string model_path,cv::Mat face_roi1,cv::Mat face_roi2)
{
    cv::Mat embeding_vector1= FaceNetInference( model_path,face_roi1);
    cv::Mat embeding_vector2= FaceNetInference( model_path,face_roi2);
    //計算 點積
    float dot=embeding_vector1.dot(embeding_vector2);
    //計算向量的歐幾里得範數
    float norm1=cv::norm(embeding_vector1);
    float norm2=cv::norm(embeding_vector2);
    // 計算餘弦相似度
    return 1- (dot / (norm1*norm2));
}


int main()
{
    cv::Mat img1=cv::imread("/home/kingargroo/cpp/facenet/test1.jpeg");
    cv::Mat img2=cv::imread("/home/kingargroo/cpp/facenet/test2.jpeg");
    std::string xml_path= "/home/kingargroo/cpp/facenet/haarcascade_frontalface_alt.xml";
    std::string model_path="/home/kingargroo/cpp/facenet/face2.onnx";
    std::vector<cv::Rect> face1,face2;
    FaceDetect(face1, xml_path,img1);
    FaceDetect(face2, xml_path,img2);
    initialize_model(model_path);
    cv::Mat face_roi1=img1(face1[0]);
    cv::Mat face_roi2=img2(face2[0]);
    float dist=calculateDistance( model_path, face_roi1, face_roi2);
    std::cout<<dist<<std::endl;
    if (dist>0.5)
    {
        std::cout<<"同一個人的概率很高"<<std::endl;
    }
    else
    {
        std::cout<<"同一個人的概率很低"<<std::endl;
    }
    

    return 0;
}
