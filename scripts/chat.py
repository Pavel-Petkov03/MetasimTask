import requests
from app.schemas import MemoryEntry, MemoryOwnerEnum
from scripts.constants import API_URL


class SalesChat:
    def __init__(self, product : str):
        self.product = product
        self.memory : list[MemoryEntry] = []

    def run(self) -> None:
        initial_response = self.__post_message(f"Product:{self.product}")
        print(f"Buyer: {initial_response["answer"]}")
        self.__add_message_to_history(initial_response["answer"], MemoryOwnerEnum.AI)
        current_input = input("You: ")
        self.__add_message_to_history(current_input, MemoryOwnerEnum.HUMAN)
        while current_input.lower() != "bye":
            response = self.__post_message(current_input)
            text_answer = response["answer"]
            self.__add_message_to_history(current_input, MemoryOwnerEnum.HUMAN)
            self.__add_message_to_history(text_answer, MemoryOwnerEnum.AI)
            print(f"Buyer: {text_answer}")
            current_input = input("You: ")


    def __post_message(self, current_message : str):

        response =  requests.post(
            f"{API_URL}/chat",
            json={
                "current_message" : current_message,
                "memory" : [entry.model_dump() for entry in self.memory],
                "product" : self.product
            }
        )
        if response.status_code != 200:
            raise RuntimeError(response.text)
        return response.json()

    def __add_message_to_history(self, message : str, role : MemoryOwnerEnum):
        if role == MemoryOwnerEnum.AI:
            self.memory.append(MemoryEntry(text=message, role=MemoryOwnerEnum.AI))
            return
        self.memory.append(MemoryEntry(text=message, role=MemoryOwnerEnum.HUMAN))


def main():
    product = input("Enter the product you want to sell.")
    try:
        chat_request = SalesChat(product)
        chat_request.run()
    except Exception as error:
        print(error)


if __name__ == "__main__":
    main()

