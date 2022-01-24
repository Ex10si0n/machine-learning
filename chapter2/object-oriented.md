# Object Oriented

Object-Oriented is also available in Python. Here is some simple code for illustrating OOP in Python.

```python
class Dog:
    species = "Canis familiaris"
    
    def __init__(self, name, age, breed):
        self.name = name
        self.age = age
        self.breed = breed

    def __str__(self):
        return f"{self.name} is {self.age} years old"

    def speak(self, sound):
        print(f"{self.name} says :{sound}")
        
if __name__ == '__main__':
    dogs = []
    dogs.append(Dog("Miles", 4, "Jack Russell Terrier"))
    dogs.append(Dog("Buddy", 9, "Dachshund"))
    dogs.append(Dog("Jack", 3, "Bulldog"))
    dogs.append(Dog("Jim", 5, "Bulldog"))
    dogs[2].speak("I am hungry!")
    print(dogs[1])
```

More about Object Oriented, please refer to COMP212 and COMP221.
