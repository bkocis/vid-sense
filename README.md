

References: 
- https://github.com/bkocis/home-surveillance-with-multimodal-llms
- https://github.com/EvolvingLMMs-Lab/LongVA
- https://arxiv.org/html/2406.16852v1
- https://github.com/calluxpore/Real-Time-Webcam-Captioning-with-LLaVA-Phi-3-and-Ollama
- https://www.samreddy.work/artificial-intelligence-and-extended-reality/image-to-captions


what is the latest development on the front of visual LLM for image understanding?

Major advancements in visual large language models (Visual LLMs) for image understanding in 2025 focus on enhancing reasoning, broadening multi-modal capabilities beyond images and text, improving efficiency, and developing versatile AI agents
. Key developments include new models with enhanced reasoning abilities and more efficient architectures, and the application of LLM knowledge to improve visual tasks. 
Models with advanced reasoning and capabilities

- Qwen3-VL: Released by Qwen AI, this model can operate on smartphones and computers, enabling automation through human-like interactions such as tapping buttons and filling forms. It also features advanced reasoning, like using street signs within images for tool-assisted navigation.
- GLM-4.5V: Developed by Z.ai, this open-source multimodal model uses a Mixture-of-Experts (MoE) architecture for superior performance with lower inference costs. It includes a "Thinking Mode" for deeper, multi-step visual reasoning and innovative 3D Rotated Positional Encoding (3D-RoPE) for better spatial understanding.
- Kimi-VL-A3B-Thinking: From Moonshot AI, this multimodal model is optimized for complex, multi-step reasoning through "long chain-of-thought" fine-tuning and reinforcement learning.
- Any-to-any models: Next-generation models like Qwen 2.5 Omni and MiniCPM-o 2.6 handle multiple input and output modalities, including vision, speech, and language, with architectures that enable natural, streaming responses. 

Enhanced architectures for efficiency 

- FastVLM: Apple's research introduced FastVLM, a model family with checkpoints available on Hugging Face, designed for efficient vision encoding. The technology allows for real-time demonstrations in the browser using WebGPU.
- SigLIP: An improved version of the CLIP image encoder, SigLIP replaces the softmax-based contrastive learning approach with a pairwise sigmoid loss, which is more easily parallelized and efficient for training.
- MobileCLIP2: A family of efficient, multi-modal image-text models designed to be used with FastVLM, which improves multi-modal reinforced training.
- LeJEPA: This method addresses self-supervised learning for computer vision by using a more stable, heuristic-free approach. 

Leveraging LLM knowledge for visual tasks 

- LLM Visual Priors: Researchers have discovered that LLMs, even when trained exclusively on text, possess latent knowledge about the visual world. This allows them to perform visual reasoning and even generate executable code for rendering scenes without seeing images. This phenomenon also enables more data-efficient visual adaptation.
- Visual Grounding with LLM-Optic: LLM-Optic enhances visual grounding—the ability to connect language to specific regions in an image—by combining an LLM's query interpretation with a multi-modal model's visual understanding. This zero-shot method works for arbitrary objects described in natural language without requiring additional training.
- Semantic Grounding with Feedback: Studies show that providing feedback can significantly improve a VLM's semantic grounding performance. Iterative feedback loops have been shown to increase accuracy by up to five percentage points, demonstrating the benefit of continuous learning. 

Emergence of multi-modal AI agents

- Multi-modal AI agents are a significant development, allowing visual LLMs to perform complex, real-world tasks beyond simple image interpretation.
- UI navigation: Several new models, including ByteDance's UI-TARS-1.5, are being developed to understand and operate graphical user interfaces, such as those on browsers and smartphones.
- Robotics and gaming: Foundation models like MAGMA-8B are designed to facilitate both UI navigation and physical interaction with the real world, opening new applications in robotics and open-world games.
- Lightweight agents: The smolagents library now includes vision-language support, allowing VLM agents to use images dynamically for tasks like GUI control. 




Last year I made  a project using LlaVa 1.6 for captioning video stream. In which direction should I look to update/upgrade my project?

Upgrading your video stream captioning project from LLaVA 1.6 requires addressing its main limitations, primarily its limited temporal understanding and potential latency
. Modern video LLMs (Vid-LLMs) and architectural improvements offer superior temporal reasoning, greater efficiency, and higher-quality, detailed captions. 
Here are the key directions for updating or upgrading your project.
Option 1: Upgrade to LLaVA-NeXT-Video 
The LLaVA project has a dedicated video version, LLaVA-NeXT-Video, that builds upon the LLaVA-1.6 foundation but is specifically trained for video tasks. This is a natural, low-effort path for updating an existing LLaVA-based project. 

- How it works: LLaVA-NeXT-Video incorporates a stronger image model and is trained on a new, high-quality video dataset.
- Implementation: You can find the model on the LLaVA-VL GitHub page. The implementation will be similar to your current LLaVA setup, but with a new model checkpoint. 

Option 2: Transition to a new video-centric LLM
For more substantial improvements, consider migrating to a model designed from the ground up for video. These models often use more advanced architectures to overcome the temporal reasoning and efficiency challenges LLaVA-based models face.

- AuroraCap: A top-performing model that uses a token merging strategy to process long video sequences efficiently without sacrificing performance.
- GLM-4.5V: An open-source, multi-modal model with a Mixture-of-Experts (MoE) architecture that offers better performance at a lower inference cost. Its "Thinking Mode" enables deeper, multi-step visual reasoning, which could lead to more detailed captions.
- Kimi-VL-A3B-Thinking: A multimodal model fine-tuned for complex, multi-step reasoning through "long chain-of-thought" and reinforcement learning, resulting in more sophisticated captioning. 

Option 3: Improve temporal reasoning and grounding
If you want to stick with your current model, you can enhance its capabilities with additional modules or improved techniques. LLaVA-based models often struggle with temporal information, as they process video as a sequence of independent frames. 

- Add a scene-change detector: An improved version of the LaViLa model pairs a short-term captioner with a scene-change detector. When a scene change occurs, the model is prompted to generate a new, more detailed "scene caption," and this information is incorporated into the caption log.
- Train a hybrid captioner: This involves finetuning a model on synthetically generated scene captions. This allows it to act as a hybrid captioner that can generate both action and scene captions, improving the system's efficiency. 

Option 4: Optimize for better performance
For real-time streaming, optimization is crucial. LLaVA-1.6 can be resource-intensive and prone to latency. 

- Token efficiency: Implement a token merging or pruning strategy. LLaVA-based models can generate an excessive number of visual tokens for video. Techniques exist to reduce this without significant performance loss, like those used in AuroraCap.
- Efficient video encoders: Switch to more efficient, specialized encoders. For example, the FastVLM family of models and MobileCLIP2 are lightweight options that can improve multi-modal training and reduce computational overhead.
- Quality enhancement: Implement a video quality enhancement method. Since video streams are often compressed, their quality can degrade. A method that uses super-resolution to enhance the quality of compressed or down-sampled frames can significantly improve captioning performance. 

Next steps for your project

- Assess limitations: Start by reviewing the limitations you've observed with LLaVA 1.6, such as caption detail, temporal accuracy, or latency.
- Benchmark new models: Compare the performance of the latest Vid-LLMs, like LLaVA-NeXT-Video, AuroraCap, or GLM-4.5V, on your specific video data. Many open-source models are available on Hugging Face.
- Experiment with hybrid approaches: Test if a combination of your existing LLaVA setup with a new temporal module (like a scene detector) can address your most significant pain points with minimal effort.
- Optimize the pipeline: Investigate optimizations like token merging or swapping out the video encoder to enhance the real-time performance of your streaming application.
