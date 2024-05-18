# SciFIBench

[**SciFIBench: Benchmarking Large Multimodal Models for Scientific Figure Interpretation**](https://arxiv.org/abs/2405.08807)

Jonathan Roberts, Kai Han, Neil Houlsby, Samuel Albanie

[[Paper](https://arxiv.org/abs/2405.08807)] [[Data](https://huggingface.co/datasets/jonathan-roberts1/SciFIBench)][[Code](#example-code)]

## Key insights:
- We use adversarial filtering and human verification to curate a challenging, high-quality 1000-question scientific figure interpretation benchmark.
- We evaluate 30 LMM, VLM and human baselines on our SciFIBench.
- GPT-4o and Gemini-Pro 1.5 are the best-performing models, outperforming some humans.
- The mean human score still outperforms all evaluated models.
- GPT-4o is significantly better than GPT-4V.
- Leveraging a strong LLM provides robust and accurate automatic evaluation.
- Varying levels of faithfulness in question answering are shown by the LMMs evaluated.

## Curation
![](images/curation.png '')

## Main results
![](images/results.png '')

## Alignment
![](images/alignment.png '')
![](images/alignment_results.png '')



## Example code

### Data
The following code can be used to download and interact with the SciFIBench dataset.
```python
from datasets import load_dataset

# load dataset
dataset = load_dataset("jonathan-roberts1/SciFIBench") # optional: set cache_dir="PATH/TO/MY/CACHE/DIR"
# figure2caption_dataset = load_dataset("jonathan-roberts1/SciFIBench", split="Figure2Caption")
# caption2figure_dataset = load_dataset("jonathan-roberts1/SciFIBench", split="Caption2Figure")
"""
DatasetDict({
    Caption2Figure: Dataset({
        features: ['ID', 'Question', 'Options', 'Answer', 'Category', 'Images'],
        num_rows: 500
    })
    Figure2Caption: Dataset({
        features: ['ID', 'Question', 'Options', 'Answer', 'Category', 'Images'],
        num_rows: 500
    })
})
"""

# select task
figure2caption_dataset = dataset['Figure2Caption']
"""
Dataset({
    features: ['ID', 'Question', 'Options', 'Answer', 'Category', 'Images'],
    num_rows: 500
})
"""

# query items
figure2caption_dataset[40] # e.g., the 41st element
"""
{'ID': 40,
 'Question': 'Which caption best matches the image?',
 'Options': ['A)  ber vs snr for fft size=2048 using ls , lmmse , lr-lmmse .',
  'B)  ber vs snr for fft size=1024 using ls , lmmse , lr-lmmse algorithms .',
  'C)  ber vs snr for fft size=512 using ls , lmmse , lr-lmmse algorithms .',
  'D)  ber vs snr for fft size=256 using ls , lmmse , lr-lmmse algorithms with a 16 qam modulation .',
  'E)  ber vs snr for a bpsk modulation .'],
 'Answer': 'D',
 'Category': 'other cs',
 'Images': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=501x431>]}
"""
```

### Figure -> Caption task inference w/ Qwen-VL-Chat via HuggingFace transformers
The following code provides an outline of an example structure for inference on SciFIBench via HuggingFace datasets. The model inference section can be replaced with inference code for other models. In this example, additional preprocessing is required to convert the PIL format of the images to filepaths for Qwen-VL-Chat.

This example leverages a strong LLM (e.g., Gemini-Pro) for automatic evaluation, code for which is provided in the next snippet.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from beartype import beartype
from datasets import load_dataset
from tqdm import tqdm
import torch
import pandas as pd


@beartype
def hf_inference(automatic_eval: bool, model_name: str = "Qwen/Qwen-VL-Chat") -> float:

    """
    Figure -> Caption task inference with Qwen-VL-Chat via HuggingFace transformers
    """

    dataset = load_dataset("jonathan-roberts1/SciFIBench", 
                        split="Figure2Caption") # optional: set cache_dir="PATH/TO/MY/CACHE/DIR"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataframe to store results
    output_df = pd.DataFrame(columns=["Question_ID", "Output", "Answer", "Correct?"])

    # Initialise model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, 
                                                 trust_remote_code=True).eval()

    # Iterate over questions
    for item in tqdm(dataset):

        question = item['Question'] # e.g., "Which caption best matches the image?"
        options = item['Options'] # ["A) Caption A...", "B) Caption B..", ..., "E) Caption E..."]
        image = item['Images'] # [PIL image]

        # format options
        options = [option + '\n' for option in options]

        # construct simple prompt
        prompt = f"{options} {question} Let's think step by step. \
            Only provide the letter of the correct caption as your answer. Answer: \n"

        # --- add model inference here ---
        # example inference with Qwen-VL-Chat

        # save image locally and pass filename to model
        img_file = 'temp.png'
        image[0].save(img_file)
        query = tokenizer.from_list_format([
            {'image': img_file}, # qwen-vl requires image filepath, not PIL
            {'text': prompt}
        ])
        response, _ = model.chat(tokenizer, query=query, history=None, do_sample=False, top_p=None, top_k=None)
        #os.remove('temp.png') # delete temp image

        if automatic_eval:
            # --- add answer extraction model here ---
            # extract answer (using gemini-1.0-pro-001)
            answer = extract_answer(response, project_id=project_id, 
                                    location=location, llm_name='gemini-1.0-pro-001')
        else:
            answer = response

        # evaluate answer
        correct = answer == item['Answer']

        results_row = {"Question_ID": item['ID'], "Output": response,
                        "Answer": answer, "Correct?": correct}
        output_df = pd.concat([output_df, pd.DataFrame([results_row])], ignore_index=True)

        # save output
        #output_df.to_csv("PATH/TO/SAVE/DIR", index=False)

    # compute accuracy
    return output_df["Correct?"].mean()

MODEL_NAME = "Qwen/Qwen-VL-Chat"
accuracy = hf_inference(automatic_eval=True, model_name=MODEL_NAME) # if True, add project_id and location
print(f"Figure -> Caption Accuracy: {100 * accuracy:.2f}%")
```


### Example automatic evaluation using Gemini-Pro 1.0 via VertexAI
Example extraction of letter answer from the potentially noisy LMM output. Gemini-Pro 1.0 can be replaced with alternative models.
```python
from beartype import beartype
import vertexai
from vertexai.preview.generative_models import GenerativeModel


@beartype
def extract_answer(lmm_output: str, project_id: str, location: str,
                   llm_name: str='gemini-1.0-pro-001') -> str:

    """ Automatically extract the answer from the (potentially noisy) output of 
    a generative multimodal model. """

    eval_prompt = ['Here is the output from a generative model:\n"'] + \
                ['"\nThe output contains the answer to a multiple choice question with options A) - E). \
                Return only the letter of the answer. If no answer is found, return "None".']
    vertexai.init(project=project_id, location=location)
    generative_multimodal_model = GenerativeModel(llm_name)
    config = {
            "max_output_tokens": 3,
            "temperature": 0,
            "top_k": 1
        }
    model_input = [eval_prompt[0] + lmm_output + eval_prompt[1]]
    return generative_multimodal_model.generate_content(contents=model_input, 
                                                        generation_config=config).text
```


### Caption -> Figure task example inference
Example inference for the Caption -> Figure task requiring interleaved text and images input. The model inference section can be replaced with inference code for other models. The images here are also converted to filepaths for preprocessing.
```python
from vertexai.preview.generative_models import Image as GCPImage # to avoid conflict with PIL.Image
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from beartype import beartype
from tqdm import tqdm
import torch
import pandas as pd
from datasets import load_dataset


@beartype
def gcp_inference(automatic_eval: bool, project_id: str, region: str,
                  model_name: str = 'gemini-pro-vision') -> float:
 
    dataset = load_dataset("jonathan-roberts1/SciFIBench", 
                        split="Caption2Figure") # optional: set cache_dir="PATH/TO/MY/CACHE/DIR"

    # dataframe to store results
    output_df = pd.DataFrame(columns=["Question_ID", "Output", "Answer", "Correct?"])

    # Initialise generative multimodal model
    vertexai.init(project=project_id, location=region)
    generative_multimodal_model = GenerativeModel(model_name)
    config = {
            "max_output_tokens": 2048,
            "temperature": 0,
            "top_k": 1
        }

    # Iterate over questions
    for item in tqdm(dataset):

        question = item['Question'] # e.g., "Caption: Caption_Text... Which image best matches the caption?"
        options = item['Options'] # ["A) image0...", "B) image1..", ..., "E) image4..."]
        images = item['Images'] # [PIL_image0, PIL_image1, ..., PIL_image4]

        # --- interleaved prompt construction and inference using GCP ---
        interleaved_prompt = []
        for prefix, img in zip(options, images):
            interleaved_prompt.append(prefix[0:3]) # extracting 'A) ', 'B) ', etc.
            img_file = 'temp.png'
            img.save(img_file)
            interleaved_prompt.append(GCPImage.load_from_file(img_file))   
        model_input = interleaved_prompt + [f"{question} Let's think step by step. \
                Only provide the letter of the correct image as your answer. Answer: \n"]
        #model_input = [img] + [question]
        response = generative_multimodal_model.generate_content(contents=model_input,
                                                                generation_config=config).text

        if automatic_eval:
            # --- add answer extraction model here ---
            # extract answer (using gemini-1.0-pro-001)
            answer = extract_answer(response, project_id=PROJECT_ID, 
                                    location=region, llm_name='gemini-1.0-pro-001')
        else:
            answer = response

        # evaluate answer
        correct = answer == item['Answer']

        # store results
        results_row = {"Question_ID": item['ID'], "Output": response,
                        "Answer": answer, "Correct?": correct}
        output_df = pd.concat([output_df, pd.DataFrame([results_row])], ignore_index=True)

        # save output
        #output_df.to_csv("PATH/TO/SAVE/DIR", index=False)

    # compute accuracy
    return output_df["Correct?"].mean()
    
PROJECT_ID = "YOUR_PROJECT_ID" # GCP project ID
LOCATION = "LOCATION" # e.g., "us-central1"
MODEL_NAME = 'gemini-pro-vision'
accuracy = gcp_inference(automatic_eval=True, project_id=PROJECT_ID,
                         region=LOCATION, model_name=MODEL_NAME)
print(f"{MODEL_NAME} Caption -> Figure Accuracy: {100 * accuracy:.2f}%")
```

