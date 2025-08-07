import os
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from scripts.constants import API_URL, LLM_CHUNK_SIZE, LLM_CHUNK_OVERLAP


class TextCleaner:
    def __init__(self, filename, result_filename="output.txt"):
        self.filename = filename
        self.result_filename = result_filename
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=LLM_CHUNK_SIZE,
            chunk_overlap=LLM_CHUNK_OVERLAP
        )

    def clean(self) -> None:
        print("Cleaning...")
        document_chunks = self.__get_documents_chunks()
        processed_chunks = self.__process_chunks(document_chunks)
        merged_text = "".join(processed_chunks)
        self.__save_to_result_file(merged_text)
        print(f"Result file: {self.result_filename}")


    @staticmethod
    def __process_chunks(chunks : list[Document]) -> list[str]:
        result = []
        for chunk in chunks:
            response = requests.post(
                f"{API_URL}/clean_text",
                json={"text" : chunk.page_content},
            )
            if response.status_code != 200:
                raise RuntimeError(response.text)
            cleared_text = response.json()["cleared_text"]
            result.append(cleared_text)
        return result


    def __save_to_result_file(self, text: str) -> None:
        with open(self.result_filename, "w") as f:
            f.write(text)

    def __get_documents_chunks(self) -> list[Document]:
        loader = TextLoader(self.filename, encoding="utf-8")
        documents = loader.load()
        return documents


if __name__ == "__main__":
    filepath = input("Enter filename: ")
    while not os.path.exists(filepath):
        print("File doesn't exist")
        filepath = input("Enter filename: ")
    output_path = input("Enter output filename[output.txt by default if left empty]")
    if output_path:
        text_cleaner = TextCleaner(filepath, output_path)
    else:
        text_cleaner = TextCleaner(filepath)
    try:
        text_cleaner.clean()
    except Exception as error:
        print(error)