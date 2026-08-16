from dataclasses import dataclass
import string


@dataclass(frozen=True)
class SimplePrompter:
    template: str

    def __post_init__(self) -> None:
        required_field_names = ["text_a", "text_b"]
        found_field_names = [
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self.template)
            if field_name is not None
        ]
        if not all(
            [
                required_field_name in found_field_names
                for required_field_name in required_field_names
            ]
        ):
            raise ValueError(
                f"Template has to contain following fields: {', '.join(required_field_names)}"
            )

    def run(self, text_a: str, text_b: str) -> str:
        return self.template.format(text_a=text_a, text_b=text_b)
