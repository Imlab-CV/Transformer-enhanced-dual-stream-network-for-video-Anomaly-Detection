<div align="justify">

## TDS-Net: Transformer enhanced dual-stream network for video Anomaly Detection

This paper has been published in [Expert Systems with Applications](https://www.sciencedirect.com/journal/expert-systems-with-applications) 

### 1. The Proposed Framework
The threat of fire is pervasive, poses significant risks to the environment, and may include potential fatalities, property devastation, and socioeconomic disruption. Successfully mitigating these risks relies on the prompt identification of fires, a process in which soft computing methodologies play a pivotal role. However, existing fire detection methods have neglected to explore the relationships among fire-indicative features, which are important to enable a model to learn more representative and robust features in remote sensing scenarios. In addition, the detection of small objects from an aerial view using unmanned arial vehicles (UAVs) or satellite imagery presents challenges in terms of capturing rich spatial detail, which reduces the model’s capabilities for accurate fire scene classification. It is also important to manage the complexity of the model effectively in order to facilitate deployment on UAVs for fast and accurate responses in emergency situations.  In response to these challenges, we propose an advanced model that integrates a modified soft attention mechanism (MSAM) and a 3D convolutional operation with a MobileNet architecture, in order to overcome obstacles related to optimising features and controlling the complexity of the model. The MSAM acts as a selective filter, enabling the model to selectively emphasise crucial features during the learning process. This adaptive attention mechanism enhances sensitivity, allowing the network to prioritise relevant details for accurate fire detection. Concurrently, the inclusion of a 3D convolutional operation extends the model's spatial awareness, enabling it to capture intricate details across multiple scales, and particularly in small regions observed from aerial viewpoints. Benchmark evaluations of the proposed model over the FD, DFAN, and ADSF datasets reveal superior performance with enhanced accuracy compared to existing methods. Our model outperforms state-of-the-art models with average improvements in accuracy of 0.54%, 2.64%, and 1.20% on the FD, DFAN, and ADSF datasets, respectively. Furthermore, the use of an explainable AI technique enhances the validation of the model visual emphasis on critical regions of the image, providing valuable insights into its decision-making process.

![](Materials/Framework.svg)
*The proposed framework:* (a) baseline model selection; (b) modified attention mechanism, where α and γ are weighting functions used to scale the important features; (c) inverted residual block; (d) depth-wise separable convolution; (e) point-wise convolution; (f) final output.

### 2. Datasets
The datasets can be downloaded from the following links.

Option 1: Download FD dataset from given link: Click here

Option 2: Download ADSF dataset from given link: Click here 

Option 3: The proposed DFAN dataset Click here


## 3. Acknowledgements
This work was supported by National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT), Grant/Award Number:(2023R1A2C1005788).
