from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageDraw
import json
import re
import os
import time


model_path = "/home/jiang/SSD_2T/pretrained/Qwen/Qwen3-VL-4B-Instruct" #too slow
print(model_path)

# default
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, dtype="auto", device_map="auto"
)

# flash_attention_2
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     model_path,
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained(model_path)

image_path = "./data/color001.jpg"

## Detect certain object
object_name = "red stirring rod"
# where = ""
where = "on the transparent rack"
text = f"I will provide you an image. Please find out the {object_name} {where}, and then output a bounding box for it, the output format is json and there are no same bounding boxs."# + "Please answer me step by step."

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path
            },
            {"type": "text", "text": text},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
start_t = time.time()
if 'Thinking' in model_path:
    generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.01)
else:
    generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.01)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
end_t = time.time()
print(output_text[0])
print(f'inference time: {end_t-start_t}s')
matches = re.findall('```json\n(.*?)\n```', output_text[0], re.DOTALL)
print('matches1:', matches)
if len(matches)==0:
    matches2 = re.findall('\[\n(.*?)\n\]', output_text[0], re.DOTALL)
    print('matches2:', matches2)
    if len(matches2)==0:
        print('Failed to extract json string.')
        exit()
    else:
        json_str = f'[{matches2[0]}]'
else:
    json_str = matches[0]

print(json_str)
bboxes = json.loads(json_str)
print(len(bboxes), bboxes)
# exit()

## show normal bbox
img = Image.open(image_path)
img_w, img_h = img.size
img1 = ImageDraw.Draw(img)
scale_x, scale_y = 1000/img_w, 1000/img_h
for bbox in bboxes:
    x1, y1, x2, y2 = bbox["bbox_2d"]
    new_x1, new_y1 = max(0, int(x1/scale_x)), max(0, int(y1/scale_y))
    new_x2, new_y2 = min(img_w-1, int(x2/scale_x)), min(img_h-1, int(y2/scale_y))
    shape = [new_x1, new_y1, new_x2, new_y2]
    print(shape)
    img1.rectangle(shape, outline ="red")
img.show()
img.save(os.path.splitext(image_path)[0].replace('data', 'results')+"_bboxes.jpg")
