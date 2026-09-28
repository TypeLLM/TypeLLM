import unittest

from typellm import TypeLLMClient


def prompts(fields):
    client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
    client.sglang.single_token = lambda label: (ord(label), label)
    return {c.name: c.opening_text() for c in client.compile_schema({"type": "object", "properties": fields})}


class UnifiedPromptTests(unittest.TestCase):
    def test_every_type_uses_field_type_instructions_answer(self):
        got = prompts({
            "item": {"type": "string", "instructions": "First item."},
            "note": {"type": ["string", "null"], "instructions": "Note."},
            "guests": {"type": "integer", "instructions": "Guests."},
            "total": {"type": ["number", "null"], "instructions": "Total."},
            "paid": {"type": "boolean", "instructions": "Paid in cash?"},
            "card": {"type": ["string", "null"], "enum": ["VISA", None], "instructions": "Card."},
        })
        self.assertEqual(got, {
            "item": 'Field: "item"\nType: string\nInstructions: First item.\n'
                    'Answer as {"item": <string>}.',
            "note": 'Field: "note"\nType: string or null\nInstructions: Note.\n'
                    'Answer as {"note": <string or null>}. Return null only if there is no value.',
            "guests": 'Field: "guests"\nType: integer\nInstructions: Guests.\n'
                      'Answer as {"guests": <integer>}.',
            "total": 'Field: "total"\nType: number or null\nInstructions: Total.\n'
                     'Answer as {"total": <number or null>}. Do not use exponent notation. '
                     'Return null only if there is no value.',
            "paid": 'Field: "paid"\nType: boolean\nInstructions: Paid in cash?\n'
                    'Choices: {"A": true, "B": false}\nAnswer as {"paid": "<label>"}.',
            "card": 'Field: "card"\nType: choice\nInstructions: Card.\n'
                    'Choices: {"A": "VISA", "B": null}\nAnswer as {"card": "<label>"}.',
        })

if __name__ == "__main__":
    unittest.main()
