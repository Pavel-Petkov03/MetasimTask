import os
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from scripts.constants import API_URL, LLM_CHUNK_SIZE, LLM_CHUNK_OVERLAP, TARGET_OUTPUT_FOLDER, TARGET_INPUT_FOLDER


class TextCleaner:
    def __init__(self, file, result_filename="output.txt"):
        self.filename = file
        self.result_filename = result_filename
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=LLM_CHUNK_SIZE,
            chunk_overlap=LLM_CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
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
        with open(TARGET_OUTPUT_FOLDER + self.result_filename, "w") as f:
            f.write(text)

    def __get_documents_chunks(self) -> list[Document]:
        loader = TextLoader(self.filename, encoding="utf-8")
        documents = loader.load()
        return self.splitter.split_documents(documents)



def main():
    filename = input("Enter filename: ")
    while not os.path.exists(TARGET_INPUT_FOLDER + filename):
        print("File doesn't exist")
        filename = input("Enter filename: ")
    final_filename = TARGET_INPUT_FOLDER + filename
    output_filename = input("Enter output filename[output.txt by default if left empty]: ")
    if output_filename:
        text_cleaner = TextCleaner(final_filename, output_filename)
    else:
        text_cleaner = TextCleaner(final_filename)
    try:
        text_cleaner.clean()
    except Exception as error:
        print(error)


if __name__ == "__main__":
    main()