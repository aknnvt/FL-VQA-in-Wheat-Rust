import os
import torch
from transformers import AutoProcessor, BlipForQuestionAnswering
from tqdm import tqdm
from PIL import Image
import random
import torch
from transformers import AutoProcessor, BlipForQuestionAnswering
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


image_paths=[]
for file in os.listdir("/home/akash/akash/vqa/vqa_images/yellow"):
  image_paths.append(os.path.join("vqa_images/yellow",file))

for file in os.listdir("/home/akash/akash/vqa/vqa_images/healthy"):
  image_paths.append(os.path.join("/home/akash/akash/vqa/vqa_images/healthy",file))

for file in os.listdir("/home/akash/akash/vqa/vqa_images/brown"):
  image_paths.append(os.path.join("/home/akash/akash/vqa/vqa_images/brown",file))

for file in os.listdir("/home/akash/akash/vqa/vqa_images/black"):
  image_paths.append(os.path.join("/home/akash/akash/vqa/vqa_images/black",file))

questions = ["What type of Wheat rust is represented in the image?",
             "At what stage plant is infected (where 0% is completely healthy to 100% at last stage off infection)?",
             "What can be the probable cause of this wheat rust?",
             "What are the environmental conditions that are favourable for the development of wheat rust as seen in an image?",
             "Based on an image of wheat rust, how can one assess the potential impact on yield?",
             "Can the presence of certain rust types in the image suggest the time of year or the wheat growth stage?",
             "How does the intensity of rust coloration in an image relate to the virulence of the infection?",
             "How does the presence of rust on the wheat spikes, as visible in an image, impact grain quality and yield?",
             "What is the significance of the size of rust pustules in determining the stage of infection?",
             "How can interveinal spaces affected by rust in an image inform about the potential for systemic infection?",
             "How might the angle and quality of light in an image affect the visibility and apparent severity of rust symptoms?",
             "What can the sharpness of the boundary between infected and healthy tissue in an image tell us about the plant's response to rust infection?",
             "How does the texture of rust pustules observed in an image relate to the stage of rust development?",
             "Can the presence of rust pustules on the lower leaves as opposed to the upper canopy, as seen in an image, indicate the timing of initial infection?",
             "What other adjacent crops are expected to be affected?"] * 40  # Repeat each question 40 times

with open('/home/akash/akash/vqa/answers.txt', 'r') as f:
    lines = f.readlines()
answers = [line.strip() for line in lines if line.strip()]

dataset = []
for i in range(len(image_paths)):
    for question, answer in zip(questions[15*i:15*(i+1)], answers[15*i:15*(i+1)]):
        image=Image.open(image_paths[i])
        horizontal_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
        vertical_flip=image.transpose(Image.FLIP_TOP_BOTTOM)
        dataset.append((image,question,answer))
        dataset.append((horizontal_flip,question,answer))
        dataset.append((vertical_flip,question,answer))

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

random.shuffle(dataset)
test_dataset=dataset[1750:1800]
dataset=dataset[0:1750]

losses = []
bleu_scores = []
test_losses = []

for epoch in range(20):
    total_loss = 0
    with tqdm(dataset, unit="batch") as tepoch:
        for i, (image, question, answer) in enumerate(tepoch):
            inputs = processor(images=image, text=question, return_tensors="pt")
            labels = processor(text=answer, return_tensors="pt").input_ids
            inputs["labels"] = labels

            inputs = {k: v.to(device) for k, v in inputs.items()}

            optimizer.zero_grad()

            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                tepoch.set_postfix({"Loss": f"{total_loss / (i + 1)}:.4f"})

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")
    losses.append(total_loss)

    test_total_loss = 0
    test_bleu_score = 0
    with torch.no_grad():
        model.eval()
        for i, (image, question, answer) in enumerate(test_dataset):
            inputs = processor(images=image, text=question, return_tensors="pt")
            labels = processor(text=answer, return_tensors="pt").input_ids
            inputs["labels"] = labels

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            loss = outputs.loss
            test_total_loss += loss.item()

            generated_answer = model.generate(**inputs)

            reference = word_tokenize(answer)
            candidate = word_tokenize(processor.decode(generated_answer, skip_special_tokens=True))
            bleu_score = sentence_bleu([reference], candidate)
            test_bleu_score += bleu_score

    test_losses.append(test_total_loss)
    test_bleu_score /= len(test_dataset)
    bleu_scores.append(test_bleu_score)
    print(f"Epoch {epoch+1}, Test Loss: {test_total_loss / len(test_dataset)}")
    print(f"Epoch {epoch+1}, Test BLEU Score: {test_bleu_score:.4f}")

model.save_pretrained('/home/akash/akash/vqa/models/blip_finetuned_bleu')

plt.plot([loss / len(dataset) for loss in dataset], label='Training Loss')
plt.plot([loss / len(test_dataset) for loss in test_losses], label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig('/home/akash/akash/vqa/loss_graph_bleu.png')
plt.show()
