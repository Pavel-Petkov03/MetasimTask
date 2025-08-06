import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
API_URL = "http://localhost:8000/"

class CleanText:
    def __init__(self, filename, result_filename="output.txt"):
        self.filename = filename
        self.result_filename = result_filename
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def clean(self) -> None:
        document_chunks = self.__get_documents_chunks()
        processed_chunks = self.__process_chunks(document_chunks)
        merged_text = "".join(processed_chunks)
        self.__save_to_result_file(merged_text)


    @staticmethod
    def __process_chunks(chunks : list[Document]) -> list[str]:
        result = []
        for chunk in chunks:
            response = requests.post(
                f"{API_URL}/clean_text",
                json={"text" : chunk.page_content},
            )
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
    try:
        text_cleaner = CleanText("../src/examples.txt")
        text_cleaner.clean()
    except Exception as error:
        print(error)