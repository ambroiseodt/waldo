for images, label_indices in dataloader:
    # Convert numeric indices to text strings
    text_labels = [my_dataset_classes[i.item()] for i in label_indices]

    # Get embeddings for both modalities
    image_features = siglip_model.get_embeddings(images)  # Your existing method
    text_features = siglip_model.get_text_embeddings(text_labels)

    # Now they are aligned and you can calculate similarity:
    # logits = image_features @ text_features.t()
