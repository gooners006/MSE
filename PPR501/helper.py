import google.generativeai as genai
from IPython.display import display, Markdown

genai.configure(api_key="AIzaSyCSuDYOuYzrk86N5KYEP442X_TSzE50GbY")
model = genai.GenerativeModel("gemini-1.5-flash")


def responsellm(prompt):
    print("Generating response...")
    response = model.generate_content(prompt)
    return display(Markdown(response.text))


def separate_code_and_markdown(response):
    """
    Separates code and markdown sections from a Jupyter Notebook text.

    Args:
        response (str): The input text containing code and markdown.

    Returns:
        dict: A dictionary with 'code' and 'markdown' keys containing the respective sections.
    """
    lines = response.split("\n")
    code_lines = []
    markdown_lines = []

    in_code_block = False

    for line in lines:
        if line.strip().startswith("```python"):
            in_code_block = True
        elif line.strip().startswith("```") and in_code_block:
            in_code_block = False
        elif in_code_block:
            code_lines.append(line)
        else:
            markdown_lines.append(line)

    return {"code": "\n".join(code_lines), "markdown": "\n".join(markdown_lines)}
