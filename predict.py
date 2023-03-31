import os
from mario_gpt import MarioLM
from cog import BasePredictor, Input, ConcatenateIterator

os.environ['TRANSFORMERS_CACHE'] = '/src/transformers-cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.mario_lm = MarioLM().to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Prompt for the Mario level"),
        seed: str = Input(description="Continue an existing level.", default=None),
        size: float = Input(
            description="Amount of columns the generated level will have", ge=0, le=1000, default=100
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        for line in self.mario_lm.sample(
            prompts=[prompt],
            seed=seed,
            num_steps=size * 14,
            temperature=2.0,
        ):
            line = self.mario_lm.tokenizer.decode(line).replace("<mask>", "Y")

            if len(line) > 1:
                yield line
