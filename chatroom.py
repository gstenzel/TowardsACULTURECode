import typing
import random
import dataclasses
import client
from rich.markup import escape
import tqdm
import json
import datetime


@dataclasses.dataclass
class Message:
    content: str
    model: str = "given"
    seed: typing.Optional[int] = None

    def __str__(self) -> str:
        return f"{self.model}: {self.content}"

    def __rstr__(self) -> str:
        return f"[red bold]{escape(self.model)}[/red bold]{escape(':')} {escape(self.content)}"


class Checks:
    @staticmethod
    def equality(chat_history: typing.List[Message]) -> bool:
        return all(
            chat_history[i].content == chat_history[i + 1].content
            for i in range(len(chat_history) - 1)
        )

    @staticmethod
    def cosine_similarity(
        chat_history: typing.List[Message], model: str, threshold: float
    ) -> bool:
        import numpy as np

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        embeddings = [client.get_embedding(model, msg.content) for msg in chat_history]
        return all(
            cosine_similarity(embeddings[i], embeddings[i + 1]) > threshold
            for i in range(len(embeddings) - 1)
        )

    @staticmethod
    def jaro_similarity(chat_history: typing.List[Message], threshold: float) -> bool:
        import jellyfish

        return all(
            jellyfish.jaro_similarity(
                chat_history[i].content, chat_history[i + 1].content
            )
            > threshold
            for i in range(len(chat_history) - 1)
        )

    @staticmethod
    def fuzzy(chat_history: typing.List[Message], threshold: float) -> bool:
        from fuzzywuzzy import fuzz

        return all(
            fuzz.ratio(chat_history[i].content, chat_history[i + 1].content) > threshold
            for i in range(len(chat_history) - 1)
        )


class ConvergenceChecker:
    def __init__(
        self,
        min_len: int = 3,
        max_step_distance: int = 5,
        to_fullfill: typing.Literal[
            "equality", "cosine_similarity", "jaro_similarity", "fuzzy"
        ] = "fuzzy",
        to_fullfill_kwargs: typing.Dict[str, typing.Any] = {},
    ):
        self.min_len = min_len
        self.max_step_distance = max_step_distance

        self._check = lambda chat_history: getattr(Checks, to_fullfill)(
            chat_history, **to_fullfill_kwargs
        )

    def __call__(self, chat_history: typing.List[Message]) -> bool:
        for step_distance in range(1, self.max_step_distance + 1):
            slice = chat_history[::-step_distance][: self.min_len]
            if len(slice) >= self.min_len:
                if self._check(slice):
                    return True
        return False


class Model:
    def __init__(self, name: str, system_prompt: str = "", **chat_kwargs):
        self.name = name
        self.chat_kwargs = chat_kwargs
        self.system_prompt = system_prompt

    def __call__(self, messages: typing.List[str]) -> Message:
        messages_ = []
        for i, message in enumerate(messages):
            messages_.append(
                dict(
                    role="system" if i % 2 == len(messages) % 2 else "user",
                    content=message,
                )
            )

        if self.system_prompt:
            messages_.insert(0, dict(role="system", content=self.system_prompt))
        return client.chat(model=self.name, messages=messages_, **self.chat_kwargs)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class Room:
    def __init__(
        self,
        models: typing.Union[typing.List[str], typing.List[Model]],
        initial_prompts: typing.Optional[typing.List[str]],
        system_prompts: typing.Optional[typing.List[str]] = [""],
        max_memory: int = 10,
        max_iterations: int = 100,
        convergence_checker: ConvergenceChecker = ConvergenceChecker(),
    ):
        if isinstance(models[0], str):
            models = [
                Model(name, system_prompt)
                for name, system_prompt in zip(models, system_prompts)
            ]
        self.max_iterations = max_iterations
        self.models = models
        self.chat_history = []
        self.max_memory = max_memory
        self.convergence_checker = convergence_checker
        for prompt in initial_prompts:
            self.chat_history.append(Message(content=prompt))

    def generate_message(self) -> typing.Generator[Message, None, None]:
        for message in self.chat_history:
            yield message
        for _ in range(self.max_iterations):
            if self.convergence_checker(self.chat_history):
                break
            model = random.choice(self.models)
            last_messages = [msg.content for msg in self.chat_history[-10:]]
            message = model(last_messages)
            self.chat_history.append(message)
            yield message
        return self.chat_history

    def run(self):
        # pbar = tqdm.tqdm(self.max_iterations, desc=f"Models: {self.models}")
        # for _ in self.generate_message():
        #     pbar.update(1)
        # pbar.close()
        with tqdm.tqdm(self.max_iterations, desc=f"Models: {self.models}") as pbar:
            for _ in self.generate_message():
                pbar.update(1)

    def to_disk(self, dir="./"):
        """
        Save the configuration of the chatroom to a json file. Also contains the chat history.
        """
        fp = (
            dir
            + "log2_"
            + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            + ".json"
        )
        print(f"Saving chatroom to {fp}")
        data = dict(
            config=dict(
                max_iterations=self.max_iterations,
                max_memory=self.max_memory,
                models=[model.name for model in self.models],
                system_prompts=[model.system_prompt for model in self.models],
            ),
            chat_history=[dataclasses.asdict(msg) for msg in self.chat_history],
        )

        with open(fp, "w") as f:
            json.dump(data, f)
