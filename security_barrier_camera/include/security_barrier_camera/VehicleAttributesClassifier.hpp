#pragma once

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include <inference_engine.hpp>
#include "security_barrier_camera/common.hpp"
#include "security_barrier_camera/ocv_common.hpp"

class VehicleAttributesClassifier {
public:
    VehicleAttributesClassifier() = default;
    VehicleAttributesClassifier(InferenceEngine::Core& ie, const std::string & deviceName,
        const std::string& xmlPath, const bool autoResize, const std::map<std::string, std::string> & pluginConfig) : ie_(ie) {
        auto network = ie.ReadNetwork(xmlPath);
        InferenceEngine::InputsDataMap attributesInputInfo(network.getInputsInfo());
        if (attributesInputInfo.size() != 1) {
            throw std::logic_error("Vehicle Attribs topology should have only one input");
        }
        InferenceEngine::InputInfo::Ptr& attributesInputInfoFirst = attributesInputInfo.begin()->second;
        attributesInputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
        if (autoResize) {
            attributesInputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
            attributesInputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);
        } else {
            attributesInputInfoFirst->setLayout(InferenceEngine::Layout::NCHW);
        }

        attributesInputName = attributesInputInfo.begin()->first;

        InferenceEngine::OutputsDataMap attributesOutputInfo(network.getOutputsInfo());
        if (attributesOutputInfo.size() != 2) {
            throw std::logic_error("Vehicle Attribs Network expects networks having two outputs");
        }
        auto it = attributesOutputInfo.begin();
        it->second->setPrecision(InferenceEngine::Precision::FP32);
        outputNameForColor = (it++)->second->getName();  // color is the first output
        it->second->setPrecision(InferenceEngine::Precision::FP32);
        outputNameForType = (it)->second->getName();  // type is the second output.

        net = ie_.LoadNetwork(network, deviceName, pluginConfig);
    }

    InferenceEngine::InferRequest createInferRequest() {
        return net.CreateInferRequest();
    }

    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect vehicleRect) {
        InferenceEngine::Blob::Ptr roiBlob = inferRequest.GetBlob(attributesInputName);
        if (InferenceEngine::Layout::NHWC == roiBlob->getTensorDesc().getLayout()) {  // autoResize is set
            InferenceEngine::ROI cropRoi{0, static_cast<size_t>(vehicleRect.x), static_cast<size_t>(vehicleRect.y), static_cast<size_t>(vehicleRect.width),
                static_cast<size_t>(vehicleRect.height)};
            InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(img);
            InferenceEngine::Blob::Ptr roiBlob = make_shared_blob(frameBlob, cropRoi);
            inferRequest.SetBlob(attributesInputName, roiBlob);
        } else {
            const cv::Mat& vehicleImage = img(vehicleRect);
            matU8ToBlob<uint8_t>(vehicleImage, roiBlob);
        }
    }
    std::pair<std::string, std::string> getResults(InferenceEngine::InferRequest& inferRequest) {
        static const std::string colors[] = {
            "white", "gray", "yellow", "red", "green", "blue", "black"
        };
        static const std::string types[] = {
            "car", "van", "truck", "bus"
        };

        // 7 possible colors for each vehicle and we should select the one with the maximum probability
        InferenceEngine::LockedMemory<const void> colorsMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(
            inferRequest.GetBlob(outputNameForColor))->rmap();
        auto colorsValues = colorsMapped.as<float*>();
        // 4 possible types for each vehicle and we should select the one with the maximum probability
        InferenceEngine::LockedMemory<const void> typesMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(
            inferRequest.GetBlob(outputNameForType))->rmap();
        auto typesValues = typesMapped.as<float*>();

        const auto color_id = std::max_element(colorsValues, colorsValues + 7) - colorsValues;
        const auto  type_id = std::max_element(typesValues,  typesValues  + 4) - typesValues;
        return std::pair<std::string, std::string>(colors[color_id], types[type_id]);
    }

private:
    std::string attributesInputName;
    std::string outputNameForColor;
    std::string outputNameForType;
    InferenceEngine::Core ie_;  // The only reason to store a device is to assure that it lives at least as long as ExecutableNetwork
    InferenceEngine::ExecutableNetwork net;
};
