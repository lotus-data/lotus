import re


def parse_cols(text: str) -> list[str]:
    # Regular expression pattern to match variables in brackets not escaped by double brackets
    pattern = r"(?<!\{)\{(?!\{)(.*?)(?<!\})\}(?!\})"
    # Find all matches in the text
    matches = re.findall(pattern, text)

    if not matches:
        raise ValueError(
            "Language expression contains no parameterized columns. Please specify the name of the relevant data column(s) in brackets {} within your language expression."
        )
    return matches


def nle2str(nle: str, cols: list[str]) -> str:
    dict = {}
    for col in cols:
        dict[col] = f"{col.capitalize()}"
    return nle.format(**dict)


# Example usage:
text = "This is a {test} string with {variable} and {{escaped_variable}}."
assert parse_cols(text) == [
    "test",
    "variable",
], f"parse_cols(text) = {parse_cols(text)}"
