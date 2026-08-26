#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>

int main()
{
    std::cout << "====================================\n";
    std::cout << "       AIRPOD YOLO DETECTOR\n";
    std::cout << "====================================\n";

    // ---------------------------------------------------------
    // Configuration
    // ---------------------------------------------------------

    const std::string MODEL_PATH = "models/airpod.onnx";

    const int INPUT_WIDTH  = 320;
    const int INPUT_HEIGHT = 320;

    // Confidence threshold
    const float CONFIDENCE_THRESHOLD = 0.50f;

    // NMS threshold
    const float NMS_THRESHOLD = 0.45f;

    // ---------------------------------------------------------
    // Load ONNX model
    // ---------------------------------------------------------

    std::cout << "Loading model...\n";
    std::cout << "Model: " << MODEL_PATH << "\n";

    cv::dnn::Net net;

    try
    {
        net = cv::dnn::readNetFromONNX(MODEL_PATH);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "\nERROR loading ONNX model:\n";
        std::cerr << e.what() << "\n\n";

        std::cerr << "Make sure this file exists:\n";
        std::cerr << MODEL_PATH << "\n";

        return 1;
    }

    if (net.empty())
    {
        std::cerr << "ERROR: Model is empty.\n";
        return 1;
    }

    // CPU
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::cout << "Model loaded successfully.\n";

    // ---------------------------------------------------------
    // Open camera
    // ---------------------------------------------------------

    std::cout << "Opening camera...\n";

    cv::VideoCapture camera(0, cv::CAP_V4L2);

    if (!camera.isOpened())
    {
        std::cerr << "ERROR: Could not open camera 0.\n";

        // Try default OpenCV backend
        camera.open(0);

        if (!camera.isOpened())
        {
            std::cerr << "Could not open camera.\n";
            return 1;
        }
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "Camera opened successfully.\n";
    std::cout << "\nPress Q or ESC to quit.\n";
    std::cout << "Press S to save a detection screenshot.\n\n";

    // ---------------------------------------------------------
    // Main loop
    // ---------------------------------------------------------

    cv::Mat frame;

    while (true)
    {
        camera >> frame;

        if (frame.empty())
        {
            std::cerr << "ERROR: Empty camera frame.\n";
            break;
        }

        // -----------------------------------------------------
        // Convert frame to YOLO input
        // -----------------------------------------------------

        cv::Mat blob;

        cv::dnn::blobFromImage(
            frame,
            blob,
            1.0 / 255.0,
            cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
            cv::Scalar(0, 0, 0),
            true,
            false
        );

        net.setInput(blob);

        // -----------------------------------------------------
        // Run inference
        // -----------------------------------------------------

        std::vector<cv::Mat> outputs;

        try
        {
            net.forward(outputs, net.getUnconnectedOutLayersNames());
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "Inference error:\n";
            std::cerr << e.what() << "\n";
            break;
        }

        if (outputs.empty())
        {
            std::cerr << "ERROR: Model returned no output.\n";
            break;
        }

        // YOLO11 exported at 320 gives:
        //
        // (1, 5, 2100)
        //
        // 5 values:
        // x, y, w, h, confidence
        //
        // Because we have only ONE class (airpod), the confidence
        // is the object confidence.

        cv::Mat output = outputs[0];

        // Convert to [2100 x 5]
        if (output.dims == 3)
        {
            const int dimensions = output.size[1];
            const int rows = output.size[2];

            output = output.reshape(1, dimensions);
            output = output.t();
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;

        // -----------------------------------------------------
        // Calculate scaling
        // -----------------------------------------------------

        float xScale =
            static_cast<float>(frame.cols) / INPUT_WIDTH;

        float yScale =
            static_cast<float>(frame.rows) / INPUT_HEIGHT;

        // -----------------------------------------------------
        // Read detections
        // -----------------------------------------------------

        for (int i = 0; i < output.rows; i++)
        {
            float x = output.at<float>(i, 0);
            float y = output.at<float>(i, 1);
            float w = output.at<float>(i, 2);
            float h = output.at<float>(i, 3);
            float confidence = output.at<float>(i, 4);

            if (confidence < CONFIDENCE_THRESHOLD)
                continue;

            // YOLO output is center x/y + width/height

            float centerX = x * xScale;
            float centerY = y * yScale;

            float width = w * xScale;
            float height = h * yScale;

            int left =
                static_cast<int>(centerX - width / 2.0f);

            int top =
                static_cast<int>(centerY - height / 2.0f);

            int boxWidth =
                static_cast<int>(width);

            int boxHeight =
                static_cast<int>(height);

            cv::Rect box(
                left,
                top,
                boxWidth,
                boxHeight
            );

            // Keep box inside image
            box &= cv::Rect(
                0,
                0,
                frame.cols,
                frame.rows
            );

            if (box.width <= 0 || box.height <= 0)
                continue;

            boxes.push_back(box);
            confidences.push_back(confidence);
        }

        // -----------------------------------------------------
        // Non-Maximum Suppression
        // -----------------------------------------------------

        std::vector<int> indices;

        cv::dnn::NMSBoxes(
            boxes,
            confidences,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD,
            indices
        );

        // -----------------------------------------------------
        // Draw detections
        // -----------------------------------------------------

        bool detected = false;
        float bestConfidence = 0.0f;

        cv::Rect bestBox;

        for (int index : indices)
        {
            detected = true;

            const cv::Rect& box = boxes[index];
            float confidence = confidences[index];

            if (confidence > bestConfidence)
            {
                bestConfidence = confidence;
                bestBox = box;
            }
        }

        if (detected)
        {
            // Draw best AirPod detection

            cv::rectangle(
                frame,
                bestBox,
                cv::Scalar(0, 255, 0),
                3
            );

            std::ostringstream label;

            label << "AirPod "
                  << std::fixed
                  << std::setprecision(1)
                  << bestConfidence * 100.0f
                  << "%";

            int baseline = 0;

            cv::Size textSize =
                cv::getTextSize(
                    label.str(),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    2,
                    &baseline
                );

            int textX = bestBox.x;

            int textY =
                std::max(
                    bestBox.y - 10,
                    textSize.height + 10
                );

            // Background for text
            cv::rectangle(
                frame,
                cv::Point(
                    textX,
                    textY - textSize.height - 10
                ),
                cv::Point(
                    textX + textSize.width + 10,
                    textY + 5
                ),
                cv::Scalar(0, 255, 0),
                cv::FILLED
            );

            cv::putText(
                frame,
                label.str(),
                cv::Point(textX + 5, textY),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 0, 0),
                2
            );

            // Status
            cv::putText(
                frame,
                "AIRPOD DETECTED",
                cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 0),
                3
            );
        }
        else
        {
            cv::putText(
                frame,
                "NO AIRPOD",
                cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 0, 255),
                3
            );
        }

        // -----------------------------------------------------
        // Show FPS-independent camera window
        // -----------------------------------------------------

        cv::imshow(
            "AirPod YOLO Detector",
            frame
        );

        // -----------------------------------------------------
        // Keyboard
        // -----------------------------------------------------

        int key = cv::waitKey(1);

        if (key == 'q' ||
            key == 'Q' ||
            key == 27)
        {
            break;
        }

        // Save screenshot
        if (key == 's' ||
            key == 'S')
        {
            static int screenshotNumber = 0;

            std::string filename =
                "airpod_detection_" +
                std::to_string(screenshotNumber++) +
                ".jpg";

            cv::imwrite(
                filename,
                frame
            );

            std::cout
                << "Saved screenshot: "
                << filename
                << "\n";
        }
    }

    // ---------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------

    camera.release();
    cv::destroyAllWindows();

    std::cout << "\nDetector stopped.\n";

    return 0;
}