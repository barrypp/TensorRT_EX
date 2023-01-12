#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "postprocess.hpp"	// postprocess plugin 
#include "logging.hpp"	
#include "calibrator.h"		// ptq
#include "json.hpp"
#include <filesystem>
#include "blockingconcurrentqueue.h"
#include <stdio.h>
#include <io.h>
#include <fcntl.h>

using namespace nvinfer1;
namespace fs = std::filesystem;
sample::Logger gLogger;

using std::cerr;
using std::endl;

// stuff we know about the network and the input/output blobs
static int INPUT_H = -1;
static int INPUT_W = -1;
static int INPUT_C = -1;
static int OUTPUT_H = -1;
static int OUTPUT_W = -1;
static int OUT_SCALE = -1;
static int OUTPUT_SIZE = -1;
static int precision_mode = -1; // fp32 : 32, fp16 : 16, int8(ptq) : 8
unsigned int maxBatchSize = 0;

const std::string INPUT_BLOB_NAME = "INPUT";
const std::string OUTPUT_BLOB_NAME = "OUTPUT";

fs::path INPUT_dir;
fs::path OUTPUT_dir;
fs::path engine_file_path;

ITensor* residualDenseBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, std::string lname);
ITensor* RRDB(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, std::string lname);

moodycamel::BlockingConcurrentQueue<std::vector<uint8_t>> load_to_proc;
moodycamel::BlockingConcurrentQueue<int> proc_to_load;
moodycamel::BlockingConcurrentQueue<std::vector<uint8_t>> proc_to_save;
moodycamel::BlockingConcurrentQueue<int> save_to_proc;

void load()
{
    unsigned int frame_length = maxBatchSize * INPUT_H * INPUT_W * INPUT_C;
    //cv::Mat img(INPUT_H, INPUT_W, CV_8UC3);
    //cv::Mat ori_img;

    int load_to_proc_size = -5;//max queue size
    int count = 0;
    while (!feof(stdin))
    {
        count++;

        std::vector<uint8_t> buf(frame_length);
        auto r = fread(buf.data(), 1, frame_length, stdin);
        if (r != frame_length)
        {
            if (feof(stdin)) break;
            if (ferror(stdin))
            {
                perror("freopen");
                exit(2);
            }
        };

        load_to_proc_size++;
        load_to_proc.enqueue(std::move(buf));
        //cerr << "load_to_proc.enqueue, " << count <<endl;

        if (load_to_proc_size > 0)
        {
            int T;
            proc_to_load.wait_dequeue(T);
            load_to_proc_size--;
        }
    }

    //cerr << "load thread, eof" << endl;
    load_to_proc.enqueue({});
}

void proc()
{
    // 2) Load engine file 
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    cerr << "===== Engine file load =====" << endl << endl;
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }
    else {
        cerr << "[ERROR] Engine file load error" << endl;
        exit(1);
    }

    // 3) Engine file deserialize
    cerr << "===== Engine file deserialize =====" << endl << endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext* context = engine->createExecutionContext();
    delete[] trtModelStream;
    void* buffers[2];
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME.c_str());
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME.c_str());

    // Allocating memory space for inputs and outputs on the GPU
    CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t)));
    CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(uint8_t)));

    // Generate CUDA stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    int proc_to_save_size = -5;//max queue size
    for (;;)
    {
        std::vector<uint8_t> input;
        std::vector<uint8_t> outputs(OUTPUT_SIZE);

        load_to_proc.wait_dequeue(input);
        proc_to_load.enqueue(0);
        if (input.size() == 0) break;

        CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        context->enqueue(maxBatchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        proc_to_save_size++;
        proc_to_save.enqueue(std::move(outputs));
        //cerr << "proc_to_save.enqueue" << std::get<1>(input) <<endl;

        if (proc_to_save_size > 0)
        {
            int T;
            save_to_proc.wait_dequeue(T);
            proc_to_save_size--;
        }
    }

    // Release stream and buffers ...
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    context->destroy();
    engine->destroy();
    runtime->destroy();

    proc_to_save.enqueue({});
}

void save()
{
    cv::Mat img(OUTPUT_H, OUTPUT_W, CV_8UC3);
    size_t length = img.total() * img.elemSize();
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    int count = 0;
    for (;;)
    {
        std::vector<uint8_t> data; ;

        proc_to_save.wait_dequeue(data);
        save_to_proc.enqueue(0);
        if (data.size() == 0) break;

        cv::Mat frame = cv::Mat(INPUT_H * OUT_SCALE, INPUT_W * OUT_SCALE, CV_8UC3, data.data());
        //cv::imshow("result", frame);
        //cv::waitKey(0);
        bool need_resize = INPUT_H * OUT_SCALE > OUTPUT_H; ;
        if (need_resize) cv::resize(frame, img, img.size(), 0, 0, cv::INTER_AREA); // resize image to output size
        size_t r = fwrite((need_resize ? img : frame).data, 1, length, stdout);
        if (r != length)
        {
            perror("TensorRT.exe : stdout broken");
            exit(3);
        }

        count++;
        if (count % 10 == 0)
        {
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
            cerr << ((double)count) / dur * 1e3 << " fps/s" << endl;
            start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            count = 0;
        }
    }
}

// Creat the engine using only the API and not any parser.
void createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, fs::path engineFile, fs::path module)
{
    cerr << "==== model build start ====" << endl << endl;
    INetworkDefinition* network = builder->createNetworkV2(0U);
    std::map<std::string, Weights> weightMap;
    weightMap = loadWeights(module.string());//not good but work
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    ITensor* data = network->addInput(INPUT_BLOB_NAME.c_str(), dt, Dims3{ INPUT_H, INPUT_W, INPUT_C });
    assert(data);

    Preprocess preprocess{ maxBatchSize, INPUT_C, INPUT_H, INPUT_W, 0 };// Custom(preprocess) plugin 
    IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");
    IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);
    IPluginV2Layer* preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);
    preprocess_layer->setName("preprocess_layer");
    ITensor* prep = preprocess_layer->getOutput(0);

    if (OUT_SCALE == 2) {
        // Pixel unshuffle.
        int h = INPUT_H / OUT_SCALE;
        int w = INPUT_W / OUT_SCALE;
        auto attn_shuffle = network->addShuffle(*prep);

        Dims shape_dims;
        std::vector<int> reshape_dims = { INPUT_C, h, OUT_SCALE, w, OUT_SCALE };
        shape_dims.nbDims = (int)reshape_dims.size();
        memcpy(shape_dims.d, reshape_dims.data(), reshape_dims.size() * sizeof(int));
        attn_shuffle->setReshapeDimensions(shape_dims);

        std::vector<int> trans_dims{ 0, 2, 4, 1, 3 };
        Permutation f_trans_dims; memcpy(f_trans_dims.order, trans_dims.data(), trans_dims.size() * sizeof(int));
        attn_shuffle->setSecondTranspose(f_trans_dims);

        auto attn_shuffle2 = network->addShuffle(*attn_shuffle->getOutput(0));
        attn_shuffle2->setReshapeDimensions(Dims3{ INPUT_C * OUT_SCALE * OUT_SCALE, h, w});
        prep = attn_shuffle2->getOutput(0);
    }

    // conv_first
    IConvolutionLayer* conv_first = network->addConvolutionNd(*prep, 64, DimsHW{ 3, 3 }, weightMap["conv_first.weight"], weightMap["conv_first.bias"]);
    conv_first->setStrideNd(DimsHW{ 1, 1 });
    conv_first->setPaddingNd(DimsHW{ 1, 1 });
    conv_first->setName("conv_first");
    ITensor* feat = conv_first->getOutput(0);

    // conv_body
    ITensor* body_feat = RRDB(network, weightMap, feat, "body.0");
    for (int idx = 1; idx < 23; idx++) {
        body_feat = RRDB(network, weightMap, body_feat, "body." + std::to_string(idx));
    }

    IConvolutionLayer* conv_body = network->addConvolutionNd(*body_feat, 64, DimsHW{ 3, 3 }, weightMap["conv_body.weight"], weightMap["conv_body.bias"]);
    conv_body->setStrideNd(DimsHW{ 1, 1 });
    conv_body->setPaddingNd(DimsHW{ 1, 1 });
    IElementWiseLayer* ew1 = network->addElementWise(*feat, *conv_body->getOutput(0), ElementWiseOperation::kSUM);
    feat = ew1->getOutput(0);

    //upsample
    IResizeLayer* interpolate_nearest = network->addResize(*feat);
    //Dims3 sclaes1 = { feat->getDimensions().d[0], feat->getDimensions().d[1]*2, feat->getDimensions().d[2]*2 };
    //interpolate_nearest->setResizeMode(ResizeMode::kNEAREST);
    //interpolate_nearest->setOutputDimensions(sclaes1);
    float sclaes1[] = { 1, 2, 2 };
    interpolate_nearest->setScales(sclaes1, 3);
    interpolate_nearest->setResizeMode(ResizeMode::kNEAREST);

    IConvolutionLayer* conv_up1 = network->addConvolutionNd(*interpolate_nearest->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap["conv_up1.weight"], weightMap["conv_up1.bias"]);
    conv_up1->setStrideNd(DimsHW{ 1, 1 });
    conv_up1->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*conv_up1->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_1->setAlpha(0.2);

    IResizeLayer* interpolate_nearest2 = network->addResize(*leaky_relu_1->getOutput(0));
    float sclaes2[] = { 1, 2, 2 };
    interpolate_nearest2->setScales(sclaes2, 3);
    interpolate_nearest2->setResizeMode(ResizeMode::kNEAREST);
    IConvolutionLayer* conv_up2 = network->addConvolutionNd(*interpolate_nearest2->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap["conv_up2.weight"], weightMap["conv_up2.bias"]);
    conv_up2->setStrideNd(DimsHW{ 1, 1 });
    conv_up2->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_2 = network->addActivation(*conv_up2->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_2->setAlpha(0.2);

    IConvolutionLayer* conv_hr = network->addConvolutionNd(*leaky_relu_2->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap["conv_hr.weight"], weightMap["conv_hr.bias"]);
    conv_hr->setStrideNd(DimsHW{ 1, 1 });
    conv_hr->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_hr = network->addActivation(*conv_hr->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_hr->setAlpha(0.2);
    IConvolutionLayer* conv_last = network->addConvolutionNd(*leaky_relu_hr->getOutput(0), 3, DimsHW{ 3, 3 }, weightMap["conv_last.weight"], weightMap["conv_last.bias"]);
    conv_last->setStrideNd(DimsHW{ 1, 1 });
    conv_last->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* out = conv_last->getOutput(0);

    //postprocess (RGB -> BGR, NCHW->NHWC, *255, ROUND, uint8)
    Postprocess postprocess{ maxBatchSize, out->getDimensions().d[0], out->getDimensions().d[1], out->getDimensions().d[2] };
    IPluginCreator* postprocess_creator = getPluginRegistry()->getPluginCreator("postprocess", "1");
    IPluginV2 *postprocess_plugin = postprocess_creator->createPlugin("postprocess_plugin", (PluginFieldCollection*)&postprocess);
    IPluginV2Layer* postprocess_layer = network->addPluginV2(&out, 1, *postprocess_plugin);
    postprocess_layer->setName("postprocess_layer");

    ITensor* final_tensor = postprocess_layer->getOutput(0);
    //show_dims(final_tensor);
    final_tensor->setName(OUTPUT_BLOB_NAME.c_str());
    network->markOutput(*final_tensor);

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1ULL << 29);  // 512MB

    if (precision_mode == 16) {
        cerr << "==== precision f16 ====" << endl << endl;
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (precision_mode == 8) {
        //cerr << "==== precision int8 ====" << endl << endl;
        //cerr << "Your platform support int8: " << builder->platformHasFastInt8() << endl;
        //assert(builder->platformHasFastInt8());
        //config->setFlag(BuilderFlag::kINT8);
        //Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(maxBatchSize, INPUT_W, INPUT_H, 0, "../data_calib/", "../Int8_calib_table/real-esrgan_int8_calib.table", INPUT_BLOB_NAME);
        //config->setInt8Calibrator(calibrator);
    }
    else {
        cerr << "==== precision f32 ====" << endl << endl;
    }

    cerr << "Building engine, please wait for a while..." << endl;
    IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
    cerr << "==== model build done ====" << endl << endl;

    cerr << "==== model selialize start ====" << endl << endl;
    std::ofstream p(engineFile, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << endl << endl;
    }
    p.write(reinterpret_cast<const char*>(engine->data()), engine->size());
    cerr << "==== model selialize done ====" << endl << endl;

    engine->destroy();
    network->destroy();
    p.close();
    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
}

int main()
{
    char* exe_path_raw[512] = {'\0'};
    ::GetModuleFileNameA(NULL, (LPSTR)exe_path_raw, 2048);
    auto exe_path = fs::path{ std::string{ (char*)exe_path_raw } }.parent_path();

    ::SetDllDirectoryA((exe_path / "vsmlrt-cuda").string().c_str());
    fs::path modulePath = exe_path / "module";
    cerr << "modulePath: " << modulePath << endl;

    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    //config
    using json = nlohmann::json;
    std::ifstream config_if("TensorRT.config.json");
    json j = json::parse(config_if,nullptr,false);
    if (j.is_discarded())
    {
        cerr << "TensorRT.exe : json::parse error" << endl;
        exit(4);
    };

    INPUT_H = j["INPUT_H"];
    INPUT_W = j["INPUT_W"];
    INPUT_C = j["INPUT_C"];
    OUTPUT_H = j["OUTPUT_H"];
    OUTPUT_W = j["OUTPUT_W"];
    OUT_SCALE = j["OUT_SCALE"];
    std::string moduleName = j["moduleName"];	// model name

    if (moduleName == "RealESRGAN_x4plus") OUT_SCALE = 4;
    else if (moduleName == "RealESRGAN_x2plus") OUT_SCALE = 2;
    else cerr << "unkown moduleName, must manual config OUT_SCALE" << endl;

    OUTPUT_SIZE = INPUT_C * INPUT_H * OUT_SCALE * INPUT_W * OUT_SCALE;
    precision_mode = j["precision_mode"];

    //
    maxBatchSize = j["maxBatchSize"];	// batch size 
    bool serialize = j["serialize"];			// TensorRT Model Serialize flag(true : generate engine, false : if no engine file, generate engine )
    
    
    char engine_file_name[256];
    sprintf(engine_file_name, "%s_%d_" "%d_%d_%d_" "%d_" ".engine", moduleName.c_str(),
        precision_mode, INPUT_H, INPUT_W, INPUT_C, OUT_SCALE); //don't know if need this much
    engine_file_path = modulePath / engine_file_name;

    // checking engine file existence
    bool exist_engine = false;
    if ((access(engine_file_path.string().c_str(), 0) != -1)) {
        exist_engine = true;
    }

    // 1) Generation engine file (decide whether to create a new engine with serialize and exist_engine variable)
    if (!((serialize == false)/*Serialize flag*/ && (exist_engine == true) /*engine existence flag*/)) {
        cerr << "===== Create Engine file =====" << endl << endl;
        IBuilder* builder = createInferBuilder(gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();
        createEngine(maxBatchSize, builder, config, DataType::kFLOAT, engine_file_path, (modulePath / moduleName).replace_extension(".wts")); // generation TensorRT Model
        builder->destroy();
        config->destroy();
        cerr << "===== Create Engine file =====" << endl << endl;
    }

    {
        cv::Mat img(OUTPUT_H, OUTPUT_W, CV_8UC3);
        if (!img.isContinuous())
        {
            cerr << "img.isContinuous() false" << endl;
            exit(3);
        }; // don't know what to do
    }

    // 4) Prepare image data for inputs
    cerr << "===== Begin img process =====" << endl << endl;

    std::thread load_thread(load);
    std::thread proc_thread(proc);
    std::thread save_thread(save);

    load_thread.join();
    proc_thread.join();
    save_thread.join();

    cerr << "TensorRT.exe ===== exit normally =====" << endl << endl;

    return 0;
}



ITensor* residualDenseBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, std::string lname)
{
    IConvolutionLayer* conv_1 = network->addConvolutionNd(*x, 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
    conv_1->setStrideNd(DimsHW{ 1, 1 });
    conv_1->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*conv_1->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_1->setAlpha(0.2);
    ITensor* x1 = leaky_relu_1->getOutput(0);

    ITensor* concat_input2[] = { x, x1 };
    IConcatenationLayer* concat2 = network->addConcatenation(concat_input2, 2);
    concat2->setAxis(0);
    IConvolutionLayer* conv_2 = network->addConvolutionNd(*concat2->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
    conv_2->setStrideNd(DimsHW{ 1, 1 });
    conv_2->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_2 = network->addActivation(*conv_2->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_2->setAlpha(0.2);
    ITensor* x2 = leaky_relu_2->getOutput(0);

    ITensor* concat_input3[] = { x, x1, x2 };
    IConcatenationLayer* concat3 = network->addConcatenation(concat_input3, 3);
    concat3->setAxis(0);
    IConvolutionLayer* conv_3 = network->addConvolutionNd(*concat3->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv3.weight"], weightMap[lname + ".conv3.bias"]);
    conv_3->setStrideNd(DimsHW{ 1, 1 });
    conv_3->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_3 = network->addActivation(*conv_3->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_3->setAlpha(0.2);
    ITensor* x3 = leaky_relu_3->getOutput(0);

    ITensor* concat_input4[] = { x, x1, x2, x3 };
    IConcatenationLayer* concat4 = network->addConcatenation(concat_input4, 4);
    concat4->setAxis(0);
    IConvolutionLayer* conv_4 = network->addConvolutionNd(*concat4->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv4.weight"], weightMap[lname + ".conv4.bias"]);
    conv_4->setStrideNd(DimsHW{ 1, 1 });
    conv_4->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_4 = network->addActivation(*conv_4->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_4->setAlpha(0.2);
    ITensor* x4 = leaky_relu_4->getOutput(0);

    ITensor* concat_input5[] = { x, x1, x2, x3, x4 };
    IConcatenationLayer* concat5 = network->addConcatenation(concat_input5, 5);
    concat5->setAxis(0);
    IConvolutionLayer* conv_5 = network->addConvolutionNd(*concat5->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap[lname + ".conv5.weight"], weightMap[lname + ".conv5.bias"]);
    conv_5->setStrideNd(DimsHW{ 1, 1 });
    conv_5->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* x5 = conv_5->getOutput(0);

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *scval = 0.2;
    Weights scale{ DataType::kFLOAT, scval, 1 };
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *shval = 0.0;
    Weights shift{ DataType::kFLOAT, shval, 1 };
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *pval = 1.0;
    Weights power{ DataType::kFLOAT, pval, 1 };

    IScaleLayer* scaled = network->addScale(*x5, ScaleMode::kUNIFORM, shift, scale, power);
    IElementWiseLayer* ew1 = network->addElementWise(*scaled->getOutput(0), *x, ElementWiseOperation::kSUM);
    return ew1->getOutput(0);
}

ITensor* RRDB(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, std::string lname)
{
    ITensor* out = residualDenseBlock(network, weightMap, x, lname + ".rdb1");
    out = residualDenseBlock(network, weightMap, out, lname + ".rdb2");
    out = residualDenseBlock(network, weightMap, out, lname + ".rdb3");

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *scval = 0.2;
    Weights scale{ DataType::kFLOAT, scval, 1 };
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *shval = 0.0;
    Weights shift{ DataType::kFLOAT, shval, 1 };
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *pval = 1.0;
    Weights power{ DataType::kFLOAT, pval, 1 };

    IScaleLayer* scaled = network->addScale(*out, ScaleMode::kUNIFORM, shift, scale, power);
    IElementWiseLayer* ew1 = network->addElementWise(*scaled->getOutput(0), *x, ElementWiseOperation::kSUM);
    return ew1->getOutput(0);
}
