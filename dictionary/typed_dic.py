from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email: str

new_person: Person = {
    "name": "raj",
    "age": 25,
    "email": "raj@example.com"
}
print(new_person)
