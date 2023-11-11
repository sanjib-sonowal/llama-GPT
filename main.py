from llama_cpp import Llama
llm = Llama(model_path="./models/ggml-vicuna-7b-1.1-q8_0.bin")


def process(transcript):
    resp = llm("Q: " + transcript + " A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
    return resp["choices"][0]["text"].split("A:")[1]


if __name__ == "__main__":
    output = process("Name the planets in the solar system?")
    print(output)
