## **Main focus**: mutimodality and varied model sizes using main transformer components (encoder and decoder); our choices are motivated by models staying close to the original Transformer architecture (https://arxiv.org/pdf/1706.03762)

### **TODO**:

- Required packages
    - torch
    - torchvision
    - numpy, matplotlib, sklearn
    - huggingface transformers
    - mteb benchmark https://github.com/embeddings-benchmark/mteb
    - openclip https://github.com/mlfoundations/open_clip 
    - cuml https://github.com/rapidsai/cuml (see https://docs.rapids.ai/install/#selector for installation)

- Add models
    - Vision 
        - ViT (base, large, huge) https://huggingface.co/google/vit-base-patch16-224 (included HF transformers)
        - Dinov2 (small, base, large, giant) https://huggingface.co/facebook/dinov2-base (included HF transformers)
        - Dinov3 (small, base, large, 7B) https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m (included HF transformers)
        - AIM (0.6B, 1.2B, 2.7B, 6.5B) https://huggingface.co/apple/AIM-600M
        - Justification: natural to do probing on those models, cite the original Dinov2/v3 and AIM papers
    - Multimodal 
        - OpenClip (base, large, huge) https://github.com/mlfoundations/open_clip 
        - SigLip (base, large) https://huggingface.co/google/siglip-base-patch16-224 (included HF transformers)
        - sigLip2 (base, large, giant) https://huggingface.co/google/siglip2-base-patch16-224 (included HF transformers)
        - Justification: natural choices, following Transferable Representation at ICLR 2024 (https://arxiv.org/pdf/2310.00927)
    - Language 
        - Llama3.1 8B https://huggingface.co/meta-llama/Llama-3.1-8B
        - Llama3.2 (1B, 3B) https://huggingface.co/meta-llama/Llama-3.2-1B  
        - Gemma3 (1B, 4B, 12B)  https://huggingface.co/google/gemma-3-4b-pt (included HF transformers)
        - Qwen3 (0.6B, 1.7B, 4B) https://huggingface.co/Qwen/Qwen3-0.6B (included HF transformers)
        - Justification: following Layer by layer (https://arxiv.org/pdf/2502.02013): generic models to study their representation and avoid potential cofounder of instruct models or models finetuned for the task of embedding
        - Although not embedding specific models, our aim is not performance-oriented but to understand the impact of layer and module on the hidden representation quality so make sense to use the general pretrained model since creating an embedding model implies extensive finetuning (see https://arxiv.org/pdf/2503.07891) and might destort the generic hidden representation. Also easier to access all the internal representations with open-source general models than API ones and embedding specific (again our aim is to understand the transformer architecture, not reach the highest performance on probing tasks). Again pretrained models encompass the general knowledge of the model while instruct-ones are laigned to prioritize helpfulness and safety. We want to probe the general knowledge so focus on non-instruct models. Same argument for general model and not embedding ones, noting that most embedding models are based on general pretrained ones after finetuning (e.g. Gemini embedding (https://arxiv.org/pdf/2503.07891) and Mistral embedding (https://arxiv.org/pdf/2401.00368))


- Add datasets
    - Vision (fine-grained classification)
        - DomainNet (5 domains)
        - ImageNet-C (5 corruptions, Contrast, Gaussian Noise, Motion Blur, Snow, Speckle Noise)
        - Cifar10
        - Cifar100
        - ImageNet
        - Pet
        - Flowers102
        - Food101 (https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Food101.html)
        - ~~StandfordCars (https://docs.pytorch.org/vision/main/generated/torchvision.datasets.StanfordCars.html)~~
        - ~~Sun397 (https://docs.pytorch.org/vision/main/generated/torchvision.datasets.SUN397.html)~~
        - Aircraft (https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FGVCAircraft.html)
        - ~~Caltech101 (https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Caltech101.html#torchvision.datasets.Caltech101)~~
        - ~~DTD (https://docs.pytorch.org/vision/main/generated/torchvision.datasets.DTD.html#torchvision.datasets.DTD)~~
        - Justification: natural choice following Dinov2, Dinov3, AIM (https://arxiv.org/pdf/2401.08541) and originaly in SimCLR (https://arxiv.org/pdf/2002.05709)
    - Language
        - MMTEB benchmarks
            - Justification: natural choice following Layer by Layer (https://arxiv.org/pdf/2502.02013) + natural choice to investigate quality of sentence embeddings (see MTEB paper https://arxiv.org/pdf/2502.13595 and https://arxiv.org/pdf/2210.07316)
            - Focus on classification tasks 
                - Justification: allows us to maintain a consistent evaluation protocol since linear probing is done with logistic regression for classification tasks in the original MTEB and MMTEB papers
            - Focus on english subset (eng v2) vs multilingual subset (following notations of Gemini embedding (https://arxiv.org/pdf/2503.07891)) 
                - Study the impact of domains and languages as distribution shifts
                - Justification: natural choice of distribution shifts, allow for diverse downstream datasets and used in prior works to study transfer of knowledge
                - Impact of domain shift has been studied in robustness of NLP models in EMNLP (https://arxiv.org/pdf/2306.00168)
                - Impact of language shift has been studied in Cross-lingual represetnation learning in ACL (https://arxiv.org/pdf/1911.02116)



- Add data preprocessing
    - Vision
        - patch_size: 14 or 16 
        - resolution: 224 or 256?
        - training data croping and so on? 
    - Language
        - Handled by the MTEB benchmark https://github.com/embeddings-benchmark/mteb

- Add experimental setup
    - Goal: consistency across modality, pretraining type, types of models; we aim for simplicity and efficiency and want to avoid confounders
    - Study transfer learning from pretraiing to downstream tasks as done in my paper on ViT finetuning (https://arxiv.org/pdf/2602.06883)
    - Model decomposition
        - Layers:
            - Loop over each block
        - Transformer block (8 outputs per block)
            - Main transformer components: ln1, mha, rc1, ln2, fc1, act, fc2, rc2
            - Architectural changes
                - Normalization layers
                    - LayerNorm
                    - RMSNorm
                - Attention module
                    - GQA
                - FFN
                    - SwiGlu activation (DiNov2,DiNov3, Llama3)
                    - Corresponding decomposition:
                        - $FC1 = self.W_in1(x)$
                        - $Act = F.silu(self.W_in1(x)) * self.W_in2(x)$
                        - $FC2 = self.W_out(Act)$
    - Linear probing 
        - Vision
            - Choice: Logistic regression with LBFGS solver with L2 regularization
            - Main motivation: DinoV2 and DinoV3 which inspire in the first place from SimCLR, ICML 2019 (https://arxiv.org/pdf/2002.05709)
            - Our implementation follows: 
            - Custom implementation for vision and done via MTEB implementation for text (standard setup in classification task)
            - Justification: following DinoV2 (https://arxiv.org/pdf/2304.07193, page 14, section 7.2), Clip (https://arxiv.org/pdf/2103.00020, page 6, section 3), Transferable Representation at ICLR 2024 (https://arxiv.org/pdf/2310.00927, page 9, table 2) and MMTEB protocal for classification (https://arxiv.org/pdf/2502.13595, page 37, section B.1) and my paper on linear probing; it allows us to avoid the impact of learning rate and batch size choice 
            - Related work: mention that attentive probing also exists (better performance in AIM) but for simplicity and avoid influence of hyperparameter, so we consider linear probing with logistic regression as done in DinoV2 and MTEB classification tasks
        - Language
            - Handled by the MTEB benchmark 
        - Efficiency to deal with large input tensors:
            - cuml.LogisticRegression is a drop-in replacement of sklearn.LogisticRegression but faster and gpu-friendly

    - Creation of embeddings
        - Choice: global average pooling over sequences of tokens without special tokens
        - Justification: consistent experimental framework to compare hidden representations across models, modality, layers and modules
        - Our goal is not performance-oriented but consistency and it allows comparing models that have CLS and those without it, and focus on global information on the content (image or text)
        - Using other probing strategy like CLS token or EOS (end-of-sentence) tokens might bias the results by the architectural objective (next-token prediction vs supervised and supervised vs discriminative self-supervised). Average pooling is neutral and avoid this bias. 
        - Also, since we compare different layers and different modules, it is important to aggregate all the information in the sequence of tokens to avoid loosing some crucial information that might be captured for instance after the attention module but not after the feedforward network, or similarly that are captured in early layers but not in the final ones. Notably, the CLS or EOS information in early layers or in some specific modules might not be valuable. Taking the average content embedding is a safer choice.
        - Vision
            - Remove CLS tokens, only consider patch tokens to remain consistent with models that don't have them
            - Justification: standard practice named GAP (global average pooling) used in Dinov2, Dinov3
        - Language
            - Remove special tokens (BOS, EOS, pad tokens), only consider the sentence embedding
            - Justification: akin to consider only patch tokens in vision
        - Related work: [interpreting physics in video models](https://arxiv.org/pdf/2602.07050), they alsoy rely on mean-pooled probe over content patches (space-time patches in their case)



    - Related work
        - iGPT paper in ICML 2020 (https://proceedings.mlr.press/v119/chen20s.html): the authors show that for pretrained generative models, middle layers give the best representation (in terms of linear probing accuracy), see Figure 2


### Memory and efficienty
- Language
    - Interesting to see different memory consumption between models of the same size depending on the KV-cache (gemma3/4 consumes less while lama3 and Qwen3 requires a lot of resources, probably because of their higher context length (to check in the original paper/implementation)).
    
### Alingment experiments on VLMs
- LAION2B is a dataset typically used for VLMs like CLIP with visual and textual descriptions. It can be used to study whether text and vision embeddings are aligned and also which one is (textual embedding recover from the text tower and visual embeddings recovered across the layers and modules). Better to used a standard dataset from the literature than my own textual descriptions.

### Results tables
- highlight the literature baseline of probing the output of the last transformer block

### Connection to matryoshka representation
- paper: https://arxiv.org/pdf/2205.13147
- The impact of embedding dimension is studied in the paper above and they provide a solution to efficiently have less or more layers in an adaptable fashion
- The proposed method ensures that all representations is useful and rich across layers
- Might be connected to our goal and our study could maybe help improve the approach, think about it
- Also inspire from this paper MRL for flops and efficiency comparison (see fig 8)

### methodological idea
- it would be interesting to see how I can merge representations for the best improvement (might be too close to other works like zeynep akata on merginf intermeidate and final layer representations)

### other works on intermediate representation
- merging intermeidate and final representation for geo-spatial localization https://openaccess.thecvf.com/content/CVPR2026/papers/Lee_Beyond_Whats_Shared_Recovering_Lost_Unique_Information_from_Intermediate_Layers_CVPR_2026_paper.pdf (interesting because good study of how different modality, task benefit from intermeidate / final representations)

### LLM probing and mechanistic interpretavility
- nice paper with interesting ideas regarding inermediate probes and depth https://arxiv.org/pdf/2605.31220
- use it to get related work + ideas + analysis from the LLM community

### Related work
- Choice of models and datasets: https://arxiv.org/pdf/2601.09322 (also connected to our methodology so very important to mention it)
- Add a similar table to described datasets as Table 2 in https://arxiv.org/pdf/2601.09322 (nb train, nb test, nb classes, imbalance class ratio)
- Interpretability:
    - https://arxiv.org/pdf/2510.08638
    - https://arxiv.org/pdf/2602.07050
    - https://arxiv.org/pdf/2602.06218
    - Thomas Fel [papers](https://scholar.google.com/citations?hl=en&user=_Rsjt9kAAAAJ&view_op=list_works&sortby=pubdate) on interpretability of vision-language models
    - [The platonic representation hypothesis](https://arxiv.org/pdf/2405.079870)
    - https://arxiv.org/pdf/2505.22785 on the dynamics of neural networks latent space
    - https://arxiv.org/pdf/2604.09885 work on probing LLMs and VLMs to understand what they "know". Might be interesting in the writing
- Nice plots and organization:
    - https://arxiv.org/pdf/2502.08606 (scaling laws)
    - https://arxiv.org/pdf/2605.01188 (scaling laws)
- Look at mechanistic interpretrability literature on VLMs, VFMs and LLMs with probing technics, SAE analysis. Some findings or design choices might be similar; our conclusions could also be read from the interpretability viewpoint.
- https://arxiv.org/pdf/2605.26097 discuss distribution shifts from pretraining with different languages
- VLM alignment:
    - look at those papres for alignement metrics and ideas
    - https://arxiv.org/pdf/2506.16895 for VLM alignement
    - https://arxiv.org/pdf/2409.19425v2 for VLM alignement too 
    - alignement of LLM representation with vision in VLMs https://openaccess.thecvf.com/content/CVPR2026F/papers/Gan_Seeing_Helps_Reasoning_in_Language_Models_CVPRF_2026_paper.pdf