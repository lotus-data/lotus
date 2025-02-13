import re
from typing import Any

import pandas as pd

import lotus
from lotus.dtype_extensions import ImageDtype
from lotus.types import SerializationFormat
from lotus.sem_ops.deepseek_utils import format_deepseek_prompt

def context_formatter(
    multimodal_data: dict[str, Any] | str,
) -> tuple[str, list[dict[str, str]]]:
    if isinstance(multimodal_data, str):
        text = multimodal_data
        image_inputs: list[dict[str, str]] = []
    elif isinstance(multimodal_data, dict):
        image_data: dict[str, str] = multimodal_data.get("image", {})
        _image_inputs: list[tuple[dict, dict]] = [
            (
                {
                    "type": "text",
                    "text": f"[{key.capitalize()}]: \n",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
            )
            for key, base64_image in image_data.items()
        ]
        image_inputs = [m for image_input in _image_inputs for m in image_input]
        text = multimodal_data["text"] or ""
    else:
        raise ValueError("multimodal_data must be a dictionary or a string")
    return text, image_inputs

def user_message_formatter(
    multimodal_data: dict[str, Any] | str,
    user_instruction_with_tag: str | None = None,
    is_deepseek: bool = False,
    base_instruction: str | None = None,
) -> dict[str, Any]:
    text, image_inputs = context_formatter(multimodal_data)
    
    # For deepseek, include instructions in user prompt and enforce <think> start
    if is_deepseek and base_instruction:
        instruction = (
            f"{base_instruction}\n\n"
            "Start your response with '<think>' to show your reasoning, "
            "then end with '</think>' and provide your final answer.\n\n"
        )
    else:
        instruction = ""
    
    if not image_inputs or len(image_inputs) == 0:
        return {
            "role": "user",
            "content": f"{instruction}Context:\n{text}\n\n{user_instruction_with_tag}",
        }
    content = [{"type": "text", "text": f"{instruction}Context:\n{text}"}] + image_inputs
    if user_instruction_with_tag:
        content.append({"type": "text", "text": f"\n\n{user_instruction_with_tag}"})
    return {
        "role": "user",
        "content": content,
    }

def extract_formatter(
    multimodal_data: dict[str, Any],
    output_cols: dict[str, str | None],
    extract_quotes: bool = True,
    strategy: str | None = None
) -> list[dict[str, str]]:
    output_col_names = list(output_cols.keys())
    # Set the description to be the key if no value is provided
    output_cols_with_desc: dict[str, str] = {col: col if desc is None else desc for col, desc in output_cols.items()}

    # Create example JSON with just the required fields
    example_json = {
        field: "example_value" for field in output_col_names
    }

    if strategy == "deepseek":
        # Create the instruction without any backslashes in f-strings
        fields_list = ", ".join(output_col_names)
        example_fields = ", ".join([f'  "{field}": "value"' for field in output_col_names])
        
        # Build instruction using concatenation instead of f-strings
        base_instruction = (
            "Your task is to extract specific fields from the given context.\n"
            "Fields to extract: " + str(output_cols_with_desc) + "\n\n"
            "Instructions:\n"
            "1. First, show your reasoning in <think> tags\n"
            "2. Then provide ONLY a valid JSON object with these exact fields:\n" +
            fields_list + "\n\n"
            "Example format:\n"
            "<think>\n"
            "Your reasoning about how you extracted each field...\n"
            "</think>\n"
            "{\n" +
            example_fields + "\n"
            "}\n\n"
            "Important: The JSON must be properly formatted and contain exactly these fields. "
            "Do not include any other text, code blocks, or explanations outside the <think> tags."
        )
    else:
        # Build instruction using concatenation instead of f-strings
        base_instruction = (
            "Your task is to extract specific fields from the given context.\n"
            "Fields to extract: " + str(output_cols_with_desc) + "\n\n"
            "Instructions:\n"
            "1. Extract the requested fields from the context\n"
            "2. Return ONLY a valid JSON object with these exact fields:\n" +
            ", ".join(output_col_names) + "\n\n"
            "Example format:\n"
            "{\n" +
            ", ".join([f'  "{field}": "value"' for field in output_col_names]) + "\n"
            "}\n\n"
            "Important: The JSON must be properly formatted and contain exactly these fields. "
            "Do not include any other text, code blocks, or explanations."
        )
    
    is_deepseek = strategy == "deepseek"
    messages = []
    
    if not is_deepseek:
        messages.append({"role": "system", "content": base_instruction})
    
    messages.append(user_message_formatter(
        multimodal_data,
        is_deepseek=is_deepseek,
        base_instruction=base_instruction if is_deepseek else None
    ))
    return messages

def filter_formatter_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]],
    examples_answer: list[bool],
    cot_reasoning: list[str],
) -> list[dict[str, str]]:
    sys_instruction = (
        "The user will provide a claim and some relevant context.\n"
        "Your job is to determine whether the claim is true for the given context.\n"
        'First give your reasoning. Then you MUST end your output with "Answer: True or False"'
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    for idx in range(len(examples_multimodal_data)):
        ex_multimodal_data = examples_multimodal_data[idx]
        ex_ans = examples_answer[idx]
        cot = cot_reasoning[idx]
        messages.extend(
            [
                user_message_formatter(ex_multimodal_data, f"Claim: {user_instruction}"),
                {
                    "role": "assistant",
                    "content": f"Reasoning:\n{cot}\n\nAnswer: {ex_ans}",
                },
            ]
        )

    messages.append(user_message_formatter(multimodal_data, f"Claim: {user_instruction}"))
    return messages

def filter_formatter_zs_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
) -> list[dict[str, str]]:
    sys_instruction = (
        "The user will provide a claim and some relevant context.\n"
        "Your job is to determine whether the claim is true for the given context.\n"
        'First give your reasoning. Then you MUST end your output with "Answer: True or False"'
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    messages.append(user_message_formatter(multimodal_data, f"Claim: {user_instruction}"))
    return messages

def filter_formatter(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answer: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
) -> list[dict[str, str]]:
    if cot_reasoning:
        assert examples_multimodal_data is not None and examples_answer is not None
        return filter_formatter_cot(
            multimodal_data, user_instruction, examples_multimodal_data, examples_answer, cot_reasoning
        )
    elif strategy == "zs-cot":
        return filter_formatter_zs_cot(multimodal_data, user_instruction)

    base_instruction = (
        "The user will provide a claim and some relevant context.\n"
        "Your job is to determine whether the claim is true for the given context.\n"
        'You must answer with a single word, "True" or "False".'
    )
    
    is_deepseek = strategy == "deepseek"
    messages = []
    
    if not is_deepseek:
        messages.append({"role": "system", "content": base_instruction})

    if examples_multimodal_data:
        assert examples_answer is not None
        assert isinstance(examples_multimodal_data, list) and isinstance(examples_answer, list)
        for i in range(len(examples_multimodal_data)):
            ex_multimodal_data = examples_multimodal_data[i]
            ex_ans = examples_answer[i]
            messages.extend(
                [
                    user_message_formatter(
                        ex_multimodal_data,
                        f"Claim: {user_instruction}",
                        is_deepseek=is_deepseek,
                        base_instruction=base_instruction if is_deepseek else None
                    ),
                    {"role": "assistant", "content": str(ex_ans)},
                ]
            )

    messages.append(user_message_formatter(
        multimodal_data,
        f"Claim: {user_instruction}",
        is_deepseek=is_deepseek,
        base_instruction=base_instruction if is_deepseek else None
    ))
    return messages

def map_formatter_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]],
    examples_answer: list[str],
    cot_reasoning: list[str],
) -> list[dict[str, str]]:
    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
        "You must give your reasoning and then your final answer"
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    for idx in range(len(examples_multimodal_data)):
        ex_df_txt = examples_multimodal_data[idx]
        ex_ans = examples_answer[idx]
        cot = cot_reasoning[idx]
        messages.extend(
            [
                user_message_formatter(ex_df_txt, f"Instruction: {user_instruction}"),
                {
                    "role": "assistant",
                    "content": f"Reasoning:\n{cot}\n\nAnswer: {ex_ans}",
                },
            ]
        )

    messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages

def map_formatter_zs_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
) -> list[dict[str, str]]:
    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
        'First give your reasoning. Then you MUST end your output with "Answer: your answer"'
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages

def map_formatter(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answer: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
) -> list[dict[str, str]]:
    if cot_reasoning:
        assert examples_multimodal_data is not None and examples_answer is not None
        return map_formatter_cot(
            multimodal_data, user_instruction, examples_multimodal_data, examples_answer, cot_reasoning
        )
    elif strategy == "zs-cot":
        return map_formatter_zs_cot(multimodal_data, user_instruction)

    is_deepseek = strategy == "deepseek"
    messages = []
    
    if is_deepseek:
        base_instruction = (
            "Your task is to answer the given instruction based on the context.\n\n"
            "Instructions:\n"
            "1. First, show your reasoning in <think> tags\n"
            "2. Then provide ONLY a single word, phrase, or number as your final answer\n\n"
            "Example format:\n"
            "<think>\n"
            "Your reasoning about how you arrived at the answer...\n"
            "</think>\n"
            "answer\n\n"
        )
    else:
        base_instruction = (
            "The user will provide an instruction and some relevant context.\n"
            "Your job is to answer the user's instruction given the context.\n"
            "Provide a single concise answer that directly responds to the instruction."
        )
        messages.append({"role": "system", "content": base_instruction})

    if examples_multimodal_data:
        assert examples_answer is not None
        for ex_df_txt, ex_ans in zip(examples_multimodal_data, examples_answer):
            messages.extend(
                [
                    user_message_formatter(
                        ex_df_txt,
                        f"Instruction: {user_instruction}",
                        is_deepseek=is_deepseek,
                        base_instruction=base_instruction if is_deepseek else None
                    ),
                    {"role": "assistant", "content": str(ex_ans)},
                ]
            )

    messages.append(user_message_formatter(
        multimodal_data,
        f"Instruction: {user_instruction}",
        is_deepseek=is_deepseek,
        base_instruction=base_instruction if is_deepseek else None
    ))
    return messages

def agg_formatter(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    strategy: str | None = None,
) -> list[dict[str, str]]:
    """
    Format instructions for aggregation operator.

    Args:
        multimodal_data (dict[str, Any]): The multimodal data to format.
        user_instruction (str): The user instruction.
        strategy (str | None): The reasoning strategy ("deepseek" or None).

    Returns:
        list[dict[str, str]]: The formatted messages.
    """
    is_deepseek = strategy == "deepseek"
    messages = []
    
    if is_deepseek:
        base_instruction = (
            "Your task is to analyze multiple documents and provide a comprehensive answer.\n\n"
            "Instructions:\n"
            "1. First, show your reasoning in <think> tags about how you analyzed the documents\n"
            "2. Then provide your final answer that combines information from all documents\n\n"
            "Example format:\n"
            "<think>\n"
            "Your reasoning about how you analyzed and combined the documents...\n"
            "</think>\n"
            "Your final comprehensive answer\n\n"
        )
    else:
        base_instruction = (
            "Your job is to provide an answer to the user's instruction given the context below from multiple documents.\n"
            "Remember that your job is to answer the user's instruction by combining all relevant information from all provided documents, into a single coherent answer.\n"
            "Do NOT copy the format of the sources! Instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
            "You have limited space to provide your answer, so be concise and to the point.\n"
        )
        messages.append({"role": "system", "content": base_instruction})

    messages.append(user_message_formatter(
        multimodal_data,
        f"Instruction: {user_instruction}",
        is_deepseek=is_deepseek,
        base_instruction=base_instruction if is_deepseek else None
    ))
    return messages

def df2text(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Formats the given DataFrame into a string containing info from cols."""

    def custom_format_row(x: pd.Series, cols: list[str]) -> str:
        return "".join([f"[{cols[i].capitalize()}]: «{x[cols[i]]}»\n" for i in range(len(cols))])

    def clean_and_escape_column_name(column_name: str) -> str:
        clean_name = re.sub(r"[^\w]", "", column_name)  # Remove spaces and special characters
        return clean_name

    # take cols that are in df
    cols = [col for col in cols if col in df.columns]
    if len(cols) == 0:
        return [""] * len(df)

    projected_df = df[cols]
    formatted_rows: list[str] = []

    if lotus.settings.serialization_format == SerializationFormat.DEFAULT:
        formatted_rows = projected_df.apply(lambda x: custom_format_row(x, cols), axis=1).tolist()
    elif lotus.settings.serialization_format == SerializationFormat.JSON:
        formatted_rows = projected_df.to_json(orient="records", lines=True).splitlines()
    elif lotus.settings.serialization_format == SerializationFormat.XML:
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError(
                "The 'lxml' library is required for XML serialization. "
                "You can install it with the following command:\n\n"
                "    pip install 'lotus-ai[xml]'"
            )
        projected_df = projected_df.rename(columns=lambda x: clean_and_escape_column_name(x))
        full_xml = projected_df.to_xml(root_name="data", row_name="row", pretty_print=False, index=False)
        root = ET.fromstring(full_xml)
        formatted_rows = [ET.tostring(row, encoding="unicode", method="xml") for row in root.findall("row")]

    return formatted_rows

def df2multimodal_info(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    """
    Formats the given DataFrame into a string containing info from cols.
    Return a list of dictionaries, each containing text and image data.
    """
    image_cols = [col for col in cols if isinstance(df[col].dtype, ImageDtype)]
    text_cols = [col for col in cols if col not in image_cols]
    text_rows = df2text(df, text_cols)
    multimodal_data = [
        {
            "text": text_rows[i],
            "image": {col.capitalize(): df[col].array.get_image(i, "base64") for col in image_cols},
        }
        for i in range(len(df))
    ]
    return multimodal_data

def merge_multimodal_info(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merges two multimodal info lists into one. Each row of first is merged with each row of second.

    Args:
        first: list of multimodal info dictionaries
        second: list of multimodal info dictionaries

    Returns:
        list of merged multimodal info dictionaries
    """
    return [
        {
            "text": f"{first[i]['text']}\n{second[j]['text']}"
            if first[i]["text"] != "" and second[j]["text"] != ""
            else first[i]["text"] + second[j]["text"],
            "image": {**first[i]["image"], **second[j]["image"]},
        }
        for i in range(len(first))
        for j in range(len(second))
    ]

def li2text(li: list[str], name: str) -> str:
    return "".join([f"[{name}] {li[i]}\n" for i in range(len(li))])
