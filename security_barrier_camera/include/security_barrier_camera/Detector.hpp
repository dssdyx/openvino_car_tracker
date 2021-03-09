#pragma once

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include <inference_engine.hpp>
#include "security_barrier_camera/common.hpp"
#include "security_barrier_camera/ocv_common.hpp"

class Detector {
public:
    struct Result {
        std::size_t label;
        float confidence;
        cv::Rect location;
    };

    static constexpr int maxProposalCount = 200;
    static constexpr int objectSize = 7;  // Output should have 7 as a last dimension"

    Detector() = default;
    Detector(InferenceEngine::Core& ie, const std::string& deviceName, const std::string& xmlPath, const std::vector<float>& detectionTresholds,
            const bool autoResize, const std::map<std::string, std::string> & pluginConfig) :
        detectionTresholds{detectionTresholds}, ie_{ie} {
        auto network = ie.ReadNetwork(xmlPath);
        InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Detector should have only one input");
        }
        InferenceEngine::InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
        if (autoResize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);
        } else {
            inputInfoFirst->setLayout(InferenceEngine::Layout::NCHW);
        }

        detectorInputBlobName = inputInfo.begin()->first;

        // ---------------------------Check outputs ------------------------------------------------------
        InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Vehicle Detection network should have only one output");
        }
        InferenceEngine::DataPtr& _output = outputInfo.begin()->second;
        const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();
        detectorOutputBlobName = outputInfo.begin()->first;
        if (maxProposalCount != outputDims[2]) {
            throw std::logic_error("unexpected ProposalCount");
        }
        if (objectSize != outputDims[3]) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }
        _output->setPrecision(InferenceEngine::Precision::FP32);

        net = ie_.LoadNetwork(network, deviceName, pluginConfig);
    }

    InferenceEngine::InferRequest createInferRequest() {
        return net.CreateInferRequest();
    }

    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img) {
        InferenceEngine::Blob::Ptr input = inferRequest.GetBlob(detectorInputBlobName);
        if (InferenceEngine::Layout::NHWC == input->getTensorDesc().getLayout()) {  // autoResize is set
            if (!img.isSubmatrix()) {
                // just wrap Mat object with Blob::Ptr without additional memory allocation
                InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(img);
                inferRequest.SetBlob(detectorInputBlobName, frameBlob);
            } else {
                throw std::logic_error("Sparse matrix are not supported");
            }
        } else {
            matU8ToBlob<uint8_t>(img, input);
        }
    }

    std::list<Result> getResults(InferenceEngine::InferRequest& inferRequest, cv::Size upscale, std::ostream* rawResults = nullptr) {
        // there is no big difference if InferReq of detector from another device is passed because the processing is the same for the same topology
        std::list<Result> results;
        InferenceEngine::LockedMemory<const void> detectorOutputBlobMapped = InferenceEngine::as<
            InferenceEngine::MemoryBlob>(inferRequest.GetBlob(detectorOutputBlobName))->rmap();
        const float * const detections = detectorOutputBlobMapped.as<float *>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];  // in case of batch
            if (image_id < 0) {  // indicates end of detections
                break;
            }
            auto label = static_cast<decltype(detectionTresholds.size())>(detections[i * objectSize + 1]);
            float confidence = detections[i * objectSize + 2];
            //dyx test
            std::cout<<"confidence: "<<confidence<<std::endl;
            //
            if (label - 1 < detectionTresholds.size() && confidence < detectionTresholds[label - 1]) {
                continue;
            }

            cv::Rect rect;
            rect.x = static_cast<int>(detections[i * objectSize + 3] * upscale.width);
            rect.y = static_cast<int>(detections[i * objectSize + 4] * upscale.height);
            rect.width = static_cast<int>(detections[i * objectSize + 5] * upscale.width) - rect.x;
            rect.height = static_cast<int>(detections[i * objectSize + 6] * upscale.height) - rect.y;
            results.push_back(Result{label, confidence, rect});

            if (rawResults) {
                *rawResults << "[" << i << "," << label << "] element, prob = " << confidence
                            << "    (" << rect.x << "," << rect.y << ")-(" << rect.width << "," << rect.height << ")" << std::endl;
            }
        }
        return results;
    }

private:
    std::vector<float> detectionTresholds;
    std::string detectorInputBlobName;
    std::string detectorOutputBlobName;
    InferenceEngine::Core ie_;  // The only reason to store a plugin as to assure that it lives at least as long as ExecutableNetwork
    InferenceEngine::ExecutableNetwork net;
};


