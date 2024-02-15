import transformers
from transformers import StoppingCriteria

tokenizer = transformers.AutoTokenizer.from_pretrained('galatolo/cerbero-7b')

class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, prompt):
        self.target_sequence = target_sequence
        self.prompt = prompt

    def __call__(self, input_ids, scores, **kwargs):
        # Convert the generated token IDs to text and remove the initial prompt
        generated_text = tokenizer.decode(input_ids[0]).replace(self.prompt, '')
        # Halt generation if the target sequence is found
        return self.target_sequence in generated_text

    def __len__(self):
        return 1

    def __iter__(self):
        yield self