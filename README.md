# TensorRT_EX

## in progress
- code refactoring
- debug(unet, custom plugin)
- performance check

## Enviroments
***
- Windows 10 laptop
- CPU i7-11375H
- GPU RTX-3060
- Visual studio 2017
- CUDA 11.1
- TensorRT 8.0.3.4 (unet, yolov5s, real-esrgan)
- TensorRT 8.2.0.6 (detr) 
- Opencv 3.4.5
***

## custom plugin 
- Layer that perform image preprocessing(NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1] (Normalize))
- plugin_ex1.cpp (plugin sample code)
- preprocess.hpp (plugin define)
- preprocess.cu (preprocessing cuda kernel function)
- Validation_py/Validation_preproc.py (Result validation with pytorch)
***

## Classification model
### vgg11 model 
- vgg11.cpp
- with preprocess plugin
- Easy-to-use structure (regenerated according to the presence or absence of engine files)
- Easier and more intuitive code structure

***

### resnet18 model
- resnet18.cpp
- 100 images from COCO val2017 dataset for PTQ calibration
- Match all results with PyTorch
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 224x224x3 image 

[//]: # (<table border="0"  width="100%">)

[//]: # (	<tbody align="center">)

[//]: # (		<tr>)

[//]: # (			<td></td>)

[//]: # (			<td><strong>Pytorch</strong></td><td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Precision</td><td>FP32</td><td>FP16</td><td>FP32</td><td>FP16</td><td>Int8&#40;PTQ&#41;</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Duration time [ms]</td>)

[//]: # (			<td>389 ms</td>)

[//]: # (			<td>330 ms</td>)

[//]: # (			<td>199 ms </td>)

[//]: # (			<td>58 ms</td>)

[//]: # (			<td>40 ms</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>FPS [f/s]</td>)

[//]: # (			<td>257 fps</td>)

[//]: # (			<td>303 fps</td>)

[//]: # (			<td>503 fps</td>)

[//]: # (			<td>1724 fps</td>)

[//]: # (			<td>2500 fps</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Memory [GB]</td>)

[//]: # (			<td>1.449 GB</td>)

[//]: # (			<td>1.421 GB</td>)

[//]: # (			<td>1.356 GB</td>)

[//]: # (			<td>0.922 GB</td>)

[//]: # (			<td>0.870 GB</td>)

[//]: # (		</tr>)

[//]: # (	</tbody>)

[//]: # (</table>)

***

## Semantic Segmentaion model
- TensorRT 8.0.3.4 (unet)
- UNet model (unet.cpp)
- additional preprocess (resize & letterbox padding) with openCV
- postprocess (model output to image)
- Match all results with PyTorch
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 512x512x3 image

[//]: # (<table border="0"  width="100%">)

[//]: # (	<tbody align="center">)

[//]: # (		<tr>)

[//]: # (			<td></td>)

[//]: # (			<td><strong>Pytorch</strong></td><td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Precision</td><td>FP32</td><td>FP16</td><td>FP32</td><td>FP16</td><td>Int8&#40;PTQ&#41;</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Duration time [ms]</td>)

[//]: # (			<td>6621 ms</td>)

[//]: # (			<td>3458 ms</td>)

[//]: # (			<td>4722 ms </td>)

[//]: # (			<td>1858 ms</td>)

[//]: # (			<td>938 ms</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>FPS [f/s]</td>)

[//]: # (			<td>15 fps</td>)

[//]: # (			<td>29 fps</td>)

[//]: # (			<td>21 fps</td>)

[//]: # (			<td>54 fps</td>)

[//]: # (			<td>107 fps</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Memory [GB]</td>)

[//]: # (			<td>3.863 GB</td>)

[//]: # (			<td>2.677 GB</td>)

[//]: # (			<td>1.600 GB</td>)

[//]: # (			<td>1.080 GB</td>)

[//]: # (			<td>1.051 GB</td>)

[//]: # (		</tr>)

[//]: # (	</tbody>)

[//]: # (</table>)

***

## Object Detection model(ViT)
- TensorRT 8.2.0.6 (detr) 
- DETR model (detr_trt.cpp) 
- additional preprocess (mean std normalization function)
- postprocess (show out detection result to the image)
- Match all results with PyTorch
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 500x500x3 image 

[//]: # (<table border="0"  width="100%">)

[//]: # (	<tbody align="center">)

[//]: # (		<tr>)

[//]: # (			<td></td>)

[//]: # (			<td><strong>Pytorch</strong></td><td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Precision</td><td>FP32</td><td>FP16</td><td>FP32</td><td>FP16</td><td>Int8&#40;PTQ&#41;</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Duration time [ms]</td>)

[//]: # (			<td>3703 ms</td>)

[//]: # (			<td>3071 ms</td>)

[//]: # (			<td>1640 ms </td>)

[//]: # (			<td>607 ms</td>)

[//]: # (			<td>530 ms</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>FPS [f/s]</td>)

[//]: # (			<td>27 fps</td>)

[//]: # (			<td>33 fps</td>)

[//]: # (			<td>61 fps</td>)

[//]: # (			<td>165 fps</td>)

[//]: # (			<td>189 fps</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Memory [GB]</td>)

[//]: # (			<td>1.563 GB</td>)

[//]: # (			<td>1.511 GB</td>)

[//]: # (			<td>1.212 GB</td>)

[//]: # (			<td>1.091 GB</td>)

[//]: # (			<td>1.005 GB</td>)

[//]: # (		</tr>)

[//]: # (	</tbody>)

[//]: # (</table>)

***

## Object Detection model
- TensorRT 8.0.3.4 (yolov5s) 
- Yolov5s model (yolov5s.cpp) 
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 640x640x3 image resized & padded

[//]: # (<table border="0"  width="100%">)

[//]: # (	<tbody align="center">)

[//]: # (		<tr>)

[//]: # (			<td></td>)

[//]: # (			<td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Precision</td><td>FP32</td><td>FP32</td><td>Int8&#40;PTQ&#41;</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Duration time [ms]</td>)

[//]: # (			<td>772 ms</td>)

[//]: # (			<td>616 ms </td>)

[//]: # (			<td>286 ms</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>FPS [f/s]</td>)

[//]: # (			<td>129 fps</td>)

[//]: # (			<td>162 fps</td>)

[//]: # (			<td>350 fps</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Memory [GB]</td>)

[//]: # (			<td>1.670 GB</td>)

[//]: # (			<td>1.359 GB</td>)

[//]: # (			<td>0.920 GB</td>)

[//]: # (		</tr>)

[//]: # (	</tbody>)

[//]: # (</table>)

***

## Super-Resolution model
- TensorRT 8.0.3.4 (Real-ESRGAN) 
- Real-ESRGAN model (real-esrgan.cpp)
- Scale up 4x (448x640x3 -> 1792x2560x3) 
- Comparison of calculation execution time of 100 iteration and GPU memory usage

[//]: # (<table border="0"  width="100%">)

[//]: # (	<tbody align="center">)

[//]: # (		<tr>)

[//]: # (			<td></td>)

[//]: # (			<td><strong>Pytorch</strong></td><td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Precision</td><td>FP32</td><td>FP16</td><td>FP32</td><td>FP16</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Duration time [ms]</td>)

[//]: # (			<td>4109 ms</td>)

[//]: # (			<td>1936 ms</td>)

[//]: # (			<td>2139 ms </td>)

[//]: # (			<td>737 ms</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>FPS [f/s]</td>)

[//]: # (			<td>0.24 fps</td>)

[//]: # (			<td>0.52 fps</td>)

[//]: # (			<td>0.47 fps</td>)

[//]: # (			<td>1.35 fps</td>)

[//]: # (		</tr>)

[//]: # (		<tr>)

[//]: # (			<td>Memory [GB]</td>)

[//]: # (			<td>5.029 GB</td>)

[//]: # (			<td>4.407 GB</td>)

[//]: # (			<td>3.807 GB</td>)

[//]: # (			<td>3.311 GB</td>)

[//]: # (		</tr>)

[//]: # (	</tbody>)

[//]: # (</table>)

***
 
## Using C TensoRT model in Python using dll
- TRT_DLL_EX : <https://github.com/yester31/TRT_DLL_EX>
***

***

## A typical TensorRT model creation sequence using TensorRT API
0. Prepare the trained model in the training framework (generate the weight file to be used in TensorRT).
1. Implement the model using the TensorRT API to match the trained model structure.
2. Extract weights from the trained model.
3. Make sure to pass the weights appropriately to each layer of the prepared TensorRT model.
4. Build and run.
5. After the TensorRT model is built, the model stream is serialized and generated as an engine file.
6. Inference by loading only the engine file in the subsequent task(if model parameters or layers are modified, re-execute the previous (4) task).
     
***

## reference   
* tensorrtx : <https://github.com/wang-xinyu/tensorrtx>
* unet : <https://github.com/milesial/Pytorch-UNet>
* detr : <https://github.com/facebookresearch/detr>
* yolov5 : <https://github.com/ultralytics/yolov5>
* real-esrgan : <https://github.com/xinntao/Real-ESRGAN>