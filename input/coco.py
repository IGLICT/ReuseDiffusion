import json

# Define paths to local dataset files
image_dir = "/workspace/diffusion/RLNoD/input/tiny_coco"
annotations_file = f"{image_dir}/annotations/captions_train2017.json"

def load_coco2017():
    with open(annotations_file, "r") as f:
        data = json.load(f)
    id_to_captions = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in id_to_captions:
            id_to_captions[image_id] = []
        id_to_captions[image_id].append(ann["caption"])

    caption_imageFile_pairs = []
    for image_id, captions in id_to_captions.items():
        filename = f"{image_id:012d}"
        caption = captions[0]
        prompt = caption.strip().lower()
        caption_imageFile_pairs.append({'prompt': prompt, 'filename': filename})
    return caption_imageFile_pairs
    