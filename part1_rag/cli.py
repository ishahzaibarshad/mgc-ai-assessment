from .rag_engine import GroundedAssistant
import sys

if __name__ == "__main__":
    assistant = GroundedAssistant("data")
    question = " ".join(sys.argv[1:]) or input("Question: ")
    result = assistant.answer(question)
    print("\nAnswer:\n" + result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"- {s['document']} | " + ", ".join(s["sections"]))
