# 1.

class Dog:
    total = 0

    def __init__(self, name, age):
        self.name = name
        self.age = age
        Dog.total += 1
        # you can also add print statements inside constructor

    def bark(self):
        print(f"{self.name} is barking")

myDog = Dog("Sally", 20)
yourDog = Dog("Jack", 30)
print(Dog.total)     # print 2(total no of Dog objects created)

print()

#2.

class Car:

    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.__odometer = 5     # private variable

    def get_descriptive_name(self):
        long_name = f"{self.make} {self.model} {self.year}"
        return long_name.title()    # title() returns a string where the first character in every word is upper case

    # creating setter and getter for odometer
    def update_odometer(self, value):
        self.__odometer = value

    def get_odometer(self):
        return self.__odometer

c = Car("Ford", "Ford", 19)
print(c.get_descriptive_name())
print(c.get_odometer())
c.update_odometer(20)
print(c.get_odometer())


print()







# INHERITANCE:

class Child:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def isStudent(self):
        return False

class Student(Child):

    def isStudent(self):
        return True


s1 = Child("Jack")
s2 = Student("Sally")
print(s1.get_name(),s1.isStudent())
print(s2.get_name(),s2.isStudent())

print()

# Single Inheritance:

class Parent:
    def func1(self):
        print("This is a function of Parent class")

class Child(Parent):
    def func2(self):
        print("This is a function of Child class")

obj = Child()
obj.func1()
obj.func2()

print()



# Multiple Inheritance

class Father:
    fathername = ""

    def father(self):
        print(self.fathername)


class Mother:
    mothername = ""

    def mother(self):
        print(self.mothername)


class Son(Father, Mother):

    def parents(self):
        print(f"Father: {self.fathername}")
        print(f"Mother: {self.mothername}")

s1 = Son()
s1.fathername = "Rayyan"
s1.mothername = "???"
s1.parents()

print()


# Multilevel Inheritance

# base class
class Grandfather:
    def __init__(self, grandfathername):
        self.grandfathername = grandfathername

# Intermediate class
class Father(Grandfather):
    def __init__(self, fathername, grandfathername):
        self.fathername = fathername

        # invoking constructor of grandfather class
        Grandfather.__init__(self, grandfathername)

# Derived class
class Daughter(Father):
    def __init__(self, daughtername, fathername, grandfathername):
        self.daughtername = daughtername

        # invoking constructor of father class
        Father.__init__(self, fathername, grandfathername)

    def print_name(self):
        print(f"GrandFather Name: {self.grandfathername}")
        print(f"Father Name: {self.fathername}")
        print(f"Daughter Name: {self.daughtername}")

d = Daughter("Ernie", "Rayy", "Ermen")
print(d.grandfathername)
d.print_name()

print()



# Hieracrchical Inheritance:

# Base Class
class Parent:
    def func1(self):
        print("This is a function in parent class")

# Derived class 1:
class Child1(Parent):
    def func2(self):
        print("This function is in Child 1")

class Child2(Parent):
    def func3(self):
        print("This function is in Child 2")

obj1 = Child1()
obj2 = Child2()
obj1.func1()
obj1.func2()
obj2.func1()
obj2.func3()

print()


# Hybrid Inheritance:

# Hybrid inheritance in Python combines multiple types of inheritance, such as single,
# multiple, or multilevel inheritance, within a program.



# super keyword:

"""
Allows us to access data of parent class. The super() keyword in inheritance is used to call a method from a parent class
within a child class. This helps in:
1. Method overriding
2. Maintaining the inheritance chain
3. Avoiding code duplication
"""


class Parent:
    def show(self):
        print('This is the parent class')

class Child(Parent):
    def show(self):
        super().show()      # calls the show method of parent class
        print("This is the child class")

child = Child()
child.show()
print()


class Mammal:
    def __init__(self, mammalName):
        print(f"{mammalName} is a warm blooded animal")

class Dog(Mammal):
    def __init__(self, name):
        print("Dog has four legs")
        super().__init__(name)

d = Dog("Spike")
print()





# ENCAPSULATION:

"""
Public: All members in a Pytho  class are public by default
Protected: Use single underscore "_" before name of members(methods and variables) to make them protected
Private: Use double underscore "__" before name of members(methods and variables) to make them private
"""

class Base:
    def __init__(self):
        self._a = 2   # protected member

class Derived(Base):
    def __init__(self):
        Base.__init__(self)
        print('Calling protected member of a base class')
        print(self._a)

obj1 = Derived()

obj2 = Base()
print(obj2._a)      # this is not allowed
# we can call protected member only inside inherited/derived classes

print()



class base:
    def __init__(self):
        self.a = "Geeks for geeks"
        self.__c = "Hell0"    # private member

class derived(base):
    def __init__(self):
        super().__init__()
        print("Calling private member of a base class")
        print(self.__c)

obj1 = base()
print(obj1.a)
# print(obj1.__c)   # this will give error
# obj2 = derived()  # this would give error since inside constructor we are using private member of base class
print()




# If you want to access and change the private variables use getters and setters methods

class Person:
    def __init__(self, name, age = 0):
        self.name = name
        self.__age = age

    def display(self):
        print(self.name)
        print(self.__age)

    def getAge(self):
        return  self.__age

    def setAge(self, age):
        self.__age = age

person = Person("Dev", 30)

# accessing using class methods
person.display()
# changing age using setter
person.setAge(35)
# accessing using getter
print(person.getAge())
print()






# ABSTRACTION

from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def move(self):
        pass

class Human(Animal):
    def move(self):
        print("I can walk and run")

class Snake(Animal):
    def move(self):
        print("I can crawl")

class Lions(Animal):
    def move(self):
        print("I can roar")

class Dog(Animal):
    def move(self):
        print("I can bark")

H = Human()
H.move()
S = Snake()
S.move()
L = Lions()
L.move()
D = Dog()
D.move()

# A = Animal()  this line will give error: Can't Instantiate abstract class Animal without an implementation for abstract method 'move'
# We cannot create objects of Abstract classes



# abstract class may contain both concrete methods and abstract methods
# abstract base class can also provide an implementation by invoking the methods via super()
from abc import ABC, abstractmethod

class R(ABC):
    def rk(self):                               # concrete method
        print("Abstract base class")
    @abstractmethod
    def k(self):                              # abstract method
        print("Abstract method")

class K(R):
    def rk(self):
        super().rk()
        super().k()
        print("Subclass")

    def k(self):   # it is compulsory to define abstract method inside Inherited class
        pass


r = K()
r.rk()
print()



# POLYMORPHISM

# ability of the function with the same name to carry different functionality altogether


# Method Overloading or Compile Time Polymorphism
def product(a,b):
    p = a * b
    print(p)

def product(a, b, c):
    p = a * b * c
    print(p)

# product(1,2)   this line shows an error.
product(1,3,4)
print()

# Python doesn't support method overloading or compile time polymorphism.
# If a class or python script has multiple methods with the same name, the method defined in the
# last will override the earlier one.




# Polymorphism with Class and Objects
class Rabbit:
    def age(self):
        print("Age of Rabbit")
    def color(self):
        print("Color of Rabbit")

class Horse:
    def age(self):
        print("Age of Horse")
    def color(self):
        print("Color of Horse")

obj1 = Rabbit()
obj2 = Horse()
for a in (obj1, obj2):
    a.age()
    a.color()
print()



# Method Overriding or Polymorphism with Inheritance
class Animal(ABC):
    def types(self):
        print("Type of Animals")
    def age(self):
        print("Age of the Animal")

class Dog(Animal):
    def age(self):
        print("Age of the Dog")

class Human(Animal):
    def age(self):
        print("Age of the Human")

obj1 = Animal()
obj2 = Dog()
obj3 = Human()

obj1.types()
obj1.age()
obj2.types()
obj2.age()
obj3.types()
obj3.age()

# The phenomenon of reimplementing a function in the derived class is known as Method Overriding

