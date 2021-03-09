#pragma once

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include <inference_engine.hpp>
#include "security_barrier_camera/common.hpp"
#include "security_barrier_camera/ocv_common.hpp"

class Lpr {
public:
    Lpr() = default;
    Lpr(InferenceEngine::Core& ie, const std::string & deviceName, const std::string& xmlPath, const bool autoResize,
        const std::map<std::string, std::string> &pluginConfig) :
        ie_{ie} {
        auto network = ie.ReadNetwork(xmlPath);

        /** LPR network should have 2 inputs (and second is just a stub) and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        InferenceEngine::InputsDataMap LprInputInfo(network.getInputsInfo());
        if (LprInputInfo.size() != 1 && LprInputInfo.size() != 2) {
            throw std::logic_error("LPR should have 1 or 2 inputs");
        }
        InferenceEngine::InputInfo::Ptr& LprInputInfoFirst = LprInputInfo.begin()->second;
        LprInputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
        if (autoResize) {
            LprInputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
            LprInputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);
        } else {
            LprInputInfoFirst->setLayout(InferenceEngine::Layout::NCHW);
        }
        LprInputName = LprInputInfo.begin()->first;
        if (LprInputInfo.size() == 2){
            //LPR model that converted from Caffe have second a stub input
            auto sequenceInput = (++LprInputInfo.begin());
            LprInputSeqName = sequenceInput->first;
        } else {
            LprInputSeqName = "";
        }

        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        InferenceEngine::OutputsDataMap LprOutputInfo(network.getOutputsInfo());
        if (LprOutputInfo.size() != 1) {
            throw std::logic_error("LPR should have 1 output");
        }
        LprOutputName = LprOutputInfo.begin()->first;
        auto lprOutputInfo = (LprOutputInfo.begin());

        // Shape of output tensor for model that converted from Caffe is [1,88,1,1], from TF [1,1,88,1]
        size_t indexOfSequenceSize = LprInputSeqName == "" ? 2 : 1;
        maxSequenceSizePerPlate = lprOutputInfo->second->getTensorDesc().getDims()[indexOfSequenceSize];

        net = ie_.LoadNetwork(network, deviceName, pluginConfig);
    }

    InferenceEngine::InferRequest createInferRequest() {
        return net.CreateInferRequest();
    }

    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect plateRect) {
        InferenceEngine::Blob::Ptr roiBlob = inferRequest.GetBlob(LprInputName);
        if (InferenceEngine::Layout::NHWC == roiBlob->getTensorDesc().getLayout()) {  // autoResize is set
            InferenceEngine::ROI cropRoi{0, static_cast<size_t>(plateRect.x), static_cast<size_t>(plateRect.y), static_cast<size_t>(plateRect.width),
                static_cast<size_t>(plateRect.height)};
            InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(img);
            InferenceEngine::Blob::Ptr roiBlob = make_shared_blob(frameBlob, cropRoi);
            inferRequest.SetBlob(LprInputName, roiBlob);
        } else {
            const cv::Mat& vehicleImage = img(plateRect);
            matU8ToBlob<uint8_t>(vehicleImage, roiBlob);
        }

        if (LprInputSeqName != "") {
            InferenceEngine::Blob::Ptr seqBlob = inferRequest.GetBlob(LprInputSeqName);
            // second input is sequence, which is some relic from the training
            // it should have the leading 0.0f and rest 1.0f
            InferenceEngine::LockedMemory<void> seqBlobMapped =
                InferenceEngine::as<InferenceEngine::MemoryBlob>(seqBlob)->wmap();
            float* blob_data = seqBlobMapped.as<float*>();
            blob_data[0] = 0.0f;
            std::fill(blob_data + 1, blob_data + seqBlob->getTensorDesc().getDims()[0], 1.0f);
        }
    }

    std::string getResults(InferenceEngine::InferRequest& inferRequest) {
        static const char *const items[] = {
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
                "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
                "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
                "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
                "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
                "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
                "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
                "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
                "<Zhejiang>", "<police>",
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                "U", "V", "W", "X", "Y", "Z"
        };
        std::string result;
        result.reserve(14u + 6u);  // the longest province name + 6 plate signs
        // up to 88 items per license plate, ended with "-1"
        InferenceEngine::LockedMemory<const void> lprOutputMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(
            inferRequest.GetBlob(LprOutputName))->rmap();
        const auto data = lprOutputMapped.as<float*>();
        for (int i = 0; i < maxSequenceSizePerPlate; i++) {
            if (data[i] == -1) {
                break;
            }
            result += items[std::size_t(data[i])];
        }
        return result;
    }

private:
    int maxSequenceSizePerPlate;
    std::string LprInputName;
    std::string LprInputSeqName;
    std::string LprOutputName;
    InferenceEngine::Core ie_;  // The only reason to store a device as to assure that it lives at least as long as ExecutableNetwork
    InferenceEngine::ExecutableNetwork net;
};
