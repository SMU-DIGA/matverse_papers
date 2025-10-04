import asyncio
import json
import os
import re
from openai import AsyncOpenAI
import pypdf
from tqdm import tqdm

import os.path as osp
from datetime import datetime
from prompt_templates import (
    prompt_template_clean,
    models_data,
    learning_methods_data,
    tasks_data,
)
from utils import parse_date, get_venue


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


def extract_ml_infos(input_file, from_scratch=False):
    contexts_path = "./contexts/"
    process_records_path = "process_records.json"
    assert os.path.exists(process_records_path)
    with open(process_records_path, "r", encoding="utf-8") as f:
        process_records = json.load(f)

    ml_infos_path = "ml_infos.json"

    if os.path.exists(ml_infos_path):
        with open(ml_infos_path, "r", encoding="utf-8") as f:
            ml_infos = json.load(f)
    else:
        ml_infos = {}

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
        if not from_scratch:
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

    batch_size = 5
    num_items_to_process = len(items_to_process)
    num_batch = (num_items_to_process // batch_size) + (
        0 if num_items_to_process % batch_size == 0 else 1
    )

    run_with_llm = False

    if run_with_llm:
        solver = Solver()
        for batch_idx in range(num_batch):
            batch_start = batch_size * batch_idx
            batch_end = min(batch_size * (batch_idx + 1), num_items_to_process)

            contents = []
            key_list = list(items_to_process.keys())
            for i in range(batch_start, batch_end):
                contents.append(
                    prompt_template_clean.replace(
                        "[PAPER TEXT]", items_to_process[key_list[i]]
                    )
                )

            try:
                results = solver.get_results(contents)

                for idx, result in enumerate(results):
                    prediction = result["response"].choices[0].message.content

                    ml_infos[key_list[batch_start + idx]] = prediction

                    process_records[key_list[batch_start + idx]]["ml_infos"] = True

                if batch_idx > 2:
                    break
            except:
                continue

    with open(ml_infos_path, "w", encoding="utf-8") as f:
        json.dump(ml_infos, f, ensure_ascii=False, indent=2)
    with open(process_records_path, "w", encoding="utf-8") as f:
        json.dump(process_records, f, ensure_ascii=False, indent=2)


def render_to_markdown_table(input_file):
    json_file_path = input_file
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {json_file_path}")

    ml_infos_path = "ml_infos.json"

    assert os.path.exists(ml_infos_path)
    with open(ml_infos_path, "r", encoding="utf-8") as f:
        ml_infos = json.load(f)

    # markdown_lines = [
    #     "| Paper ID | Datasets | Tasks | Models | Learning Methods | Performance Highlights | Application Domains |",
    #     "|----------|----------|-------|--------|------------------|------------------------|---------------------|",
    # ]

    markdown_lines = ["## 📑 ML Infos in Papers (Chronological Order)\n"]

    # Extract items
    items = data.get("items", [])

    print("number of items {}".format(len(items)))

    # Filter journal articles and sort by date (newest first)
    papers = []
    for item in items:
        # if item.get("itemType") == "journalArticle":
        date_obj, date_str = parse_date(item.get("date", ""))
        papers.append({"item": item, "date_obj": date_obj, "date_str": date_str})

    # Sort by date (newest first)
    papers.sort(key=lambda x: x["date_obj"], reverse=True)

    for paper_id, paper in enumerate(papers):
        # Parse the nested JSON string if needed

        if paper["item"]["key"] not in ml_infos:
            continue

        try:
            paper_info = json.loads(ml_infos[paper["item"]["key"]])

            # Extract dataset names (one per line)
            datasets = [d["name"] for d in paper_info.get("datasets", [])]
            # datasets_str = "<br>".join(datasets) if datasets else "-"

            # Extract task names (one per line)
            tasks = [t["name"] for t in paper_info.get("tasks_addressed", [])]
            # tasks_str = "<br>".join(tasks) if tasks else "-"

            # Extract model names (one per line)
            models = [m["name"] for m in paper_info.get("models_used", [])]
            # models_str = "<br>".join(models) if models else "-"

            # Extract learning method names (one per line)
            learning_methods = [
                lm["name"] for lm in paper_info.get("learning_methods_used", [])
            ]
            # learning_str = "<br>".join(learning_methods) if learning_methods else "-"

            # Extract performance highlights (one per line)
            performances = []
            for combo in paper_info.get("model_task_learning_combinations", []):
                perf = combo.get("performance", {})
                metrics = perf.get("metrics", {})
                if metrics:
                    perf_items = [f"{k}: {v}" for k, v in metrics.items()]
                    performances.extend(perf_items)

            # performance_str = "<br>".join(performances) if performances else "-"
            # performance_str = ''

            def join_with_comma(items):
                """adding comma for the items not last"""
                if not items:
                    return "_None_"
                if len(items) == 1:
                    return items[0]
                #
                formatted_items = [f"{item}," for item in items[:-1]]
                formatted_items.append(items[-1])  #
                return "<br>".join(formatted_items)

            # Extract application domains (one per line)
            domains = paper_info.get("application_domains", [])

            item = paper["item"]
            # venue = item.get("publicationTitle", "Unknown Journal")
            venue = get_venue(paper)
            title = item.get("title", "Untitled")
            paper_number = (
                len(papers) - paper_id
            )  # Since papers are sorted newest first

            # index_lines.append(
            #     f"- [{paper_number}. {title}]({anchor}), {venue} *({paper['date_str']})*"
            # )
            doi = item.get("DOI", "")
            anchor = f"https://doi.org/{doi}"

            markdown_line = ""
            markdown_line += f"### [{paper_number}. {title}]({anchor}), {venue} *({paper['date_str']})*\n\n"
            markdown_line += "| Category | Items |\n"
            markdown_line += "|----------|-------|\n"
            markdown_line += f"| **Datasets** | {join_with_comma(datasets)} |\n"
            markdown_line += f"| **Models** | {join_with_comma(models)} |\n"
            markdown_line += f"| **Tasks** | {join_with_comma(tasks)} |\n"
            markdown_line += (
                f"| **Learning Methods** | {join_with_comma(learning_methods)} |\n"
            )
            markdown_line += (
                f"| **Performance Highlights** | {join_with_comma(performances)} |\n"
            )
            markdown_line += (
                f"| **Application Domains** | {join_with_comma(domains)} |\n\n"
            )
            markdown_line += "---\n\n"

            markdown_lines.append(markdown_line)

        except:
            continue

    return markdown_lines


def render_ml_taxonomy():
    """Generate ML Taxonomy with Category and Items columns"""

    # Models with categories

    def create_table(data, title):
        lines = [f"\n### {title}\n", "| Category | Items |", "|----------|-------|"]

        for category, items in data:
            items_str = ", ".join(items)
            lines.append(f"| **{category}** | {items_str} |")

        # total_items = sum(len(items) for _, items in data)
        # lines.append(f"\n**Total: {total_items} items**\n")
        return "\n".join(lines)

    # Generate full taxonomy
    output = ["## 📈 Machine Learning Taxonomy\n"]
    # output.append(
    #     "**A comprehensive classification of ML models, learning methods, and tasks**\n"
    # )

    total_models = sum(len(items) for _, items in models_data)
    total_methods = sum(len(items) for _, items in learning_methods_data)
    total_tasks = sum(len(items) for _, items in tasks_data)

    # output.append(
    #     f"\n- **{len(models_data)} Model Categories** → {total_models} specific models"
    # )
    # output.append(
    #     f"\n- **{len(learning_methods_data)} Learning Method Categories** → {total_methods} specific methods"
    # )
    # output.append(
    #     f"\n- **{len(tasks_data)} Task Categories** → {total_tasks} specific tasks\n"
    # )
    # output.append("\n### The ML Framework")
    # output.append("\n```")
    # output.append("\nML Solution = Model × Learning Method × Task")
    # output.append("\n              (What)   (How)            (Why)")
    # output.append("\n```")

    output.append("\n<div align=\"center\">\n")
    output.append("\n```\n")
    output.append("╔════════════════════════════════════════════════════════════════╗")
    output.append("║                                                                ║")
    output.append("║        ML Solution = Model × Method × Task                     ║")
    output.append("║                      (What)  (How)   (Why)                     ║")
    output.append("║                                                                ║")
    output.append("╚════════════════════════════════════════════════════════════════╝")
    output.append("\n```")
    output.append("\n</div>\n")

    output.append(
        create_table(
            models_data,
            f"📊 Table 1: Models (What to Use) [{len(models_data)} Categories → {total_models} Specifics]",
        )
    )
    output.append("\n---\n")
    output.append(
        create_table(
            learning_methods_data,
            f"🎓 Table 2: Learning Methods (How to Learn) [{len(learning_methods_data)} Categories → {total_methods} Specifics]",
        )
    )
    output.append("\n---\n")
    output.append(
        create_table(
            tasks_data,
            f"🎯 Table 3: Tasks (What to Solve) [{len(tasks_data)} Categories → {total_tasks} Specifics]",
        )
    )

    # Summary

    output.append("\n---\n")

    return output


def export_to_markdown(output_file_path, output_contents):
    current_time = datetime.now()
    markdown_lines = [
        f"""<div align="center">
    <h1>Machine Learning Infos in AI4(M)S Papers</h1> 
    <h3>Update Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}</h3>
    </div>\n\n---\n""",
        # f"**Generation Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n---\n",
        # "This is a summary of the ML information in the AI4(M)S Papers.\n",
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
        + render_ml_taxonomy()
        + output_contents
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

    extract_ml_infos(input_file)

    output_contents = render_to_markdown_table(input_file)

    export_to_markdown(output_file_path=output_file, output_contents=output_contents)
    print("Processing completed successfully!")


if __name__ == "__main__":
    set_api_keys()
    main()
