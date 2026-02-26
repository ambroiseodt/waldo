- Add models
    - Vision 
        - ViT (base, large, huge)
        - Dinov2, Dinov3
        - AIM (3 or 4 sizes)
        - Justification: natural to do probing on those models, cite the original Dinov2/v3 and AIM papers
    - Multimodal 
        - Clip
        - SigLip
        - Justification: (natural choices, following UNDERSTANDING TRANSFERABLE REPRESENTATION LEARNING AND ZERO-SHOT TRANSFER IN CLIP in ICLR 2024)
    - Text 
        - Llama 3 (1B, 3B, 7B)
        - Gemma 3 (2B, 4B, 9B)
        - Qwen 3 (3 sizes)
        - Justification: following Layer by layer (https://arxiv.org/pdf/2502.02013): generic models to study their representation and avoid potential cofounder of instruct models or models finetuned for the task of embedding

- Add datasets
    - Vision
        - Cifar10/100
        - ImageNet
        - ImageNet-C
        - Pet
        - Flowers102
        - DomainNet (5 domains)
    - Text
        - MMTEB benchmarks
            - Classification tasks (multilingual)
            - Justification: following Layer by Layer (https://arxiv.org/pdf/2502.02013)

- Experimental setup
    - Goal: consistency and simplicity for the sake of efficiency and to avoid cofounders
    - Following 

    
- Add linear classifier
    - Choice: Logistic regression with LBFGS solver 
    - Justification: following DinoV2 (https://arxiv.org/pdf/2304.07193, page 14, section 7.2) and MMTEB (https://arxiv.org/pdf/2502.13595, page 37, section B.1)
    
    to avoid the impact of learning rate and batch size choice and for simplicity (following the MTEB evaluation protocol + my paper on linear probing + Clip paper)

- Add probing strategy (unified?)
    - We want the same strategy for different types of models and some of them don't have CLS token so we propose to simply average over the sequence of tokens. Since the CLS contains valuable information (following vit original paper and Dinov2), we still consider it in the sequence of tokens for ViT and Dino models. For decoder-only models, we use the same strategy to maintain a unified setup except we don't consider the padding tokens.
    - Why is it needed: we want to compare supervised, autoregressive, self-supervised models for classification, multimodel and text so we need a unified hidden representation. Since not all models have CLS token, CLS option is not viable. But it still contains valuable information for supervised models so it makes sense to use it as part of the model representation. To capture the global representation, we propose to take the average over the sequence of tokens. It leads to a unifided strategy for ViT with CLS token (ViT, Dino) but also for autoregressive models like AIM that don't have a CLS token
    - Vision: for unified evaluation among supervised, autoregressive and self-supervised, take the global average over the sequence
        - Average the sequence of embeddings 
        - Justification: following Dinov2, patch tokens contain critical information 
    - Multimodel
        - Average the sequence of embeddings
    - Text