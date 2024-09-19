import open_clip

def list_openclip_models():
    for (model_name, pretrained) in open_clip.list_pretrained():
        print(f'{model_name} & {pretrained}')

def create_model_tokenizer(model_name, pretrained):
    # Model, Tokenizer and Transform methods
    model, _, transform = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    print(f"model: {model_name}, pretrained: {pretrained}")

    return model, tokenizer, transform


if __name__ == '__main__':
    list_openclip_models()
