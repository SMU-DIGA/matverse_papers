import asyncio
import json
import os
import re
from openai import AsyncOpenAI
import pypdf
from tqdm import tqdm

import os.path as osp
from datetime import datetime

# from npsolver import MODELS

MODELS = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    # "o1-mini": "o1-mini-2024-09-12",
    # "o3-mini": "o3-mini-2025-01-31",
    # "deepseek-chat": "deepseek/deepseek-chat",
    # "claude": "anthropic/claude-3-7-sonnet-20250219",
    "deepseek-r1": "deepseek-r1-250120",
    "deepseek-r1-32b": "deepseek-r1-distill-qwen-32b-250120",
    "deepseek-v3": "deepseek-v3-241226",
    "deepseek-v3-2503": "deepseek-v3-250324",
    # "maas-deepseek-r1": "MaaS-DS-R1",
    # "maas-deepseek-v3": "MaaS-DS-V3",
    "openrouter-mistral-3.2-24b": "mistralai/mistral-small-3.2-24b-instruct",
    "openrouter-gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite",
    "openrouter-gemini-2.5-pro": "google/gemini-2.5-pro",
    "openrouter-gemini-2.5-flash": "google/gemini-2.5-flash",
    "openrouter-k2": "moonshotai/kimi-k2",
    "openrouter-qwen3-32b": "qwen/qwen3-32b",
    "openrouter-qwen3-235b": "qwen/qwen3-235b-a22b",
    "openrouter-qwen3-32b-free": "qwen/qwen3-32b:free",
    "openrouter-deepseek-v3": "deepseek/deepseek-chat-v3-0324",
    "openrouter-claude-sonnet-4": "anthropic/claude-sonnet-4",
    "zhipuai-glm-4.5": "glm-4.5",
    "zhipuai-glm-4.5-air": "glm-4.5-air",
}


def set_api_keys():
    import os

    def load_key(file_path: str):
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return f.read().strip()
        return None

    # Required keys
    openai_key = load_key("api_keys/openai_api_key.txt")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    openrouter_key = load_key("api_keys/openrouter_api_key.txt")
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key


prompt_template = """You are an expert at extracting structured information from scientific papers in the AI for Science domain. Your task is to analyze the provided paper text and extract ALL AI-related information in JSON format, including multiple models, datasets, and their performances.

## Task: Extract AI Components from AI4Science Paper

Analyze the paper and extract ALL instances of models, datasets, and performances. Output your response as a valid JSON object with this exact structure:

{
  "datasets": [
    {
      "name": "dataset name",
      "domain": "scientific domain",
      "size": "dataset size/scale/number of samples",
      "source": "dataset source or reference",
      "preprocessing": "preprocessing methods",
      "split": "train/val/test split if mentioned",
      "description": "brief description of dataset content"
    }
  ],
  "models": [
    {
      "model_name": "specific model name",
      "model_type": "type (e.g., CNN, Transformer, GNN, VAE, Diffusion)",
      "architecture_details": "detailed architecture description",
      "backbone": "backbone network if applicable",
      "parameters": "number of parameters if mentioned",
      "pretrained": "yes/no and pretrained source",
      "novel_contribution": "what's new about this model",
      "purpose": "baseline/proposed/comparison model"
    }
  ],
  "algorithms_and_methods": {
    "training_algorithms": ["list all training approaches used"],
    "optimization_methods": ["list all optimizers"],
    "loss_functions": [
      {
        "name": "loss function name",
        "description": "brief description or formula"
      }
    ],
    "special_techniques": ["all techniques like augmentation, regularization, etc."],
    "evaluation_metrics": ["all metrics used"]
  },
  "experiments": [
    {
      "experiment_name": "name or description of experiment",
      "dataset_used": "which dataset(s)",
      "models_compared": ["list of models in this experiment"],
      "task": "specific task (e.g., classification, regression, generation)",
      "results": [
        {
          "model_name": "model name",
          "dataset": "dataset name",
          "metrics": {
            "metric_name_1": "value",
            "metric_name_2": "value"
          },
          "additional_notes": "any important notes about results"
        }
      ]
    }
  ],
  "performance_summary": [
    {
      "model": "model name",
      "dataset": "dataset name",
      "task": "task description",
      "best_metric": "name of primary metric",
      "best_score": "score value",
      "comparison": "compared to baseline/sota",
      "improvement": "improvement percentage if mentioned"
    }
  ],
  "computational_details": {
    "hardware": "all hardware mentioned",
    "training_time": "training duration per model if specified",
    "inference_time": "inference speed if mentioned",
    "hyperparameters": [
      {
        "model": "model name",
        "learning_rate": "value",
        "batch_size": "value",
        "epochs": "value",
        "other_params": "other hyperparameters"
      }
    ]
  },
  "scientific_application": {
    "domain": "scientific field",
    "problem": "specific problem addressed",
    "ai_application": "how AI is applied to solve it",
    "key_findings": ["list of main findings"],
    "limitations": "limitations mentioned if any",
    "future_work": "future directions if mentioned"
  },
  "code_and_resources": {
    "code_available": "yes/no/url",
    "pretrained_models": "availability and links",
    "supplementary": "supplementary materials mentioned"
  }
}

## Instructions:
1. Extract ALL models mentioned in the paper (proposed, baselines, comparisons)
2. Extract ALL datasets used or mentioned
3. Extract ALL performance results, experiments, and comparisons
4. Capture performance metrics for each model-dataset combination
5. Use "not_specified" for unavailable information
6. Use empty array [] for list fields with no information
7. Ensure valid JSON format
8. Be comprehensive - don't miss any model or dataset mentioned
9. Include ablation studies and variant models if present

---
[PAPER TEXT]
---

Output only the JSON object, no additional text.
"""


class Solver:
    def __init__(self, model_name="gpt-5-mini"):
        assert model_name in MODELS.keys()

        # self.args = args

        self.model_name = model_name
        # self.problem_name = problem_name

        self.local_llm = None
        # self.sampling_params = None

        self.client = None
        if model_name.startswith("gpt"):
            self.client = AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                # base_url="https://ark.cn-beijing.volces.com/api/v3",
            )

        if model_name.startswith("openrouter"):
            # print("THIS IS FOR OPENROUTER")
            # print(os.environ.get("OPENROUTER_API_KEY"))
            self.client = AsyncOpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )

    async def async_evaluate_llm(self, contents):
        assert self.client is not None

        async def call_gpt(prompt):
            response = await self.client.chat.completions.create(
                model=MODELS[self.model_name],
                messages=[{"role": "user", "content": prompt}],
            )
            return response

        return await asyncio.gather(*[call_gpt(content) for content in contents])

    def get_results(self, contents):
        try:
            print("Starting the batch calling of LLM")
            assert self.client is not None
            responses = asyncio.run(self.async_evaluate_llm(contents))
            # print(responses)
            print("End of calling LLM")
            outputs = []
            for idx, response in enumerate(responses):
                # print(response)
                token_numbers = {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                }

                output = {
                    "response": response,
                    "tokens": token_numbers,
                    "error_msg": {"llm": None, "json": None},
                }
                # print(result)
                outputs.append(output)
            return outputs
        except Exception as e:
            # return None
            # print(e)
            outputs = [
                {
                    "response": None,
                    "tokens": {"prompt": 0, "completion": 0},
                    "error_msg": {"llm": f"LLM error: {e}", "json": None},
                }
                for _ in range(len(contents))
            ]

            return outputs


def extract_pdf_path_from_zotero_item(zotero_item):
    if "attachments" in zotero_item:
        for attachment in zotero_item["attachments"]:
            if (
                attachment.get("itemType") == "attachment"
                and "pdf" in attachment.get("title", "").lower()
                and "path" in attachment
            ):
                return attachment["path"]
    return None


def read_pdf_with_pypdf(pdf_path):
    """
    Read PDF content using PyPDF2.

    Args:
        pdf_path (str): Path to PDF file

    Returns:
        str: Extracted text content
    """
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            text_content = ""

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page.extract_text()

        return text_content
    except Exception as e:
        print(f"Error reading PDF with PyPDF2: {e}")
        return None


def extract_ml_infos(input_file):
    contexts_path = "./contexts/"
    process_records_path = "process_records.json"
    assert os.path.exists(process_records_path)
    with open(process_records_path, "r", encoding="utf-8") as f:
        process_records = json.load(f)

    json_file_path = input_file
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {json_file_path}")

    items = data.get("items", [])

    items_to_process = {}
    for zotero_item in items:
        if (
            zotero_item["key"] in process_records
            and ("ml_infos" in process_records[zotero_item["key"]])
            and process_records[zotero_item["key"]]["ml_infos"]
        ):
            continue

        if (
            zotero_item["key"] in process_records
            and process_records[zotero_item["key"]]["pdf_extract"]
        ):
            txt_path = osp.join(contexts_path, zotero_item["key"] + ".txt")
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()

            items_to_process[zotero_item["key"]] = content

    batch_size = 1
    num_items_to_process = len(items_to_process)
    num_batch = (num_items_to_process // batch_size) + (
        0 if num_items_to_process % batch_size == 0 else 1
    )

    run_llm = False

    if run_llm:

        solver = Solver()
        for batch_idx in range(num_batch):
            batch_start = batch_size * batch_idx
            batch_end = min(batch_size * (batch_idx + 1), num_items_to_process)

            contents = []
            key_list = list(items_to_process.keys())
            for i in range(batch_start, batch_end):
                contents.append(
                    prompt_template.replace("[PAPER TEXT]", items_to_process[key_list[i]])
                )

            results = solver.get_results(contents)

            for result in results:
                prediction = result["response"].choices[0].message.content

                print(prediction)

            break

    else:
        return None


def export_to_markdown(output_file_path, output_contents):
    current_time = datetime.now()
    markdown_lines = [
        f"""<div align="center">
    <h1>ML Infos</h1> 
    <h3>Update Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}</h3>
    </div>\n\n---\n""",
        # f"**Generation Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n---\n",
        "This is a summary of the ML information in the AI4(M)S Papers.\n",
    ]

    markdown_lines = (
        [
            """---
layout: default
title: ML Infos
permalink: /ml_infos/
---
            """
        ]
        + markdown_lines
    )

    # Join all lines
    markdown_content = "\n".join(markdown_lines)

    # Save to file if output path is provided
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Markdown file saved to: {output_file_path}")

    return markdown_content


def extract_context_from_pdf(input_file, specified_items=None):
    contexts_path = "./contexts/"
    process_records_path = "process_records.json"
    if os.path.exists(process_records_path):
        with open(process_records_path, "r", encoding="utf-8") as f:
            process_records = json.load(f)
    else:
        process_records = {}
        with open(process_records_path, "w", encoding="utf-8") as f:
            json.dump(process_records, f, ensure_ascii=False, indent=2)

    json_file_path = input_file
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {json_file_path}")

    items = data.get("items", [])

    print("number of items {}".format(len(items)))

    # Filter journal articles and sort by date (newest first)
    # for item in tqdm(data_list, desc="处理数据"):

    for zotero_item in tqdm(items, desc="extracting contents from pdf files"):
        try:
            if not specified_items:
                if (
                    zotero_item["key"] in process_records
                    and process_records[zotero_item["key"]]["pdf_extract"]
                ):
                    continue
            else:
                if zotero_item["key"] not in specified_items:
                    continue
            pdf_path = extract_pdf_path_from_zotero_item(zotero_item)

            # print(pdf_path)

            if pdf_path:
                content = read_pdf_with_pypdf(pdf_path)

                txt_path = osp.join(contexts_path, zotero_item["key"] + ".txt")

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(content)

                    f.close()
                process_records[zotero_item["key"]] = {"pdf_extract": True}
        except:
            print("error when process: {}".format(zotero_item["key"]))

            continue

    with open(process_records_path, "w", encoding="utf-8") as f:
        json.dump(process_records, f, ensure_ascii=False, indent=2)


def main():
    """
    Main function to demonstrate usage
    """
    # Example usage
    input_file = "AI4S.json"  # Replace with your JSON file path
    output_file = "ml_infos.md"  # Output markdown file

    extract_context_from_pdf(input_file)

    output_contents = extract_ml_infos(input_file)

    export_to_markdown(output_file_path=output_file, output_contents=output_contents)
    print("Processing completed successfully!")


if __name__ == "__main__":
    set_api_keys()
    main()
