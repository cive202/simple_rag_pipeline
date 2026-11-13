import pymupdf4llm
from pathlib import Path


class ToMarkdown:
    def __init__(self, pdf_path: Path | str):
        self.pdf_path = Path(pdf_path)
        self.file_type = self.pdf_path.suffix.lower()

    def pdf2md(self) -> str:
        pdf_path = self.pdf_path

        if not pdf_path.exists():
            raise FileNotFoundError("File not found")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only .pdf and txt files are supported")
        print(f"Converting {pdf_path}")

        markdown = pymupdf4llm.to_markdown(str(pdf_path))

        if not markdown.strip():
            raise ValueError("no extractable text found in pdf")

        return markdown

    def txt2text(self, txt_path: Path | str) -> str:
        txt_path = Path(txt_path)

        if not txt_path.exists():
            raise FileNotFoundError(f"File not found: {txt_path}")

        if not txt_path.is_file():
            raise ValueError("Path must point to a file")

        if txt_path.suffix.lower() != ".txt":
            raise ValueError("Only TXT files are supported")

        text = txt_path.read_text(encoding="utf-8", errors="replace")

        if not text.strip():
            raise ValueError("Document contains no text")

        return text


if __name__ == "__main__":
    path = Path(input("give me the path"))
    converter = ToMarkdown(path)
    if path.suffix.lower() == ".pdf":
        print(converter.pdf2md())
    if path.suffix.lower() == ".txt":
        print(converter.txt2text(path))
