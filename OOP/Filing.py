# 1. Reading from a Text File
with open('text.txt') as f:
    content = f.read()   # reads the entire content of the file as a single string
print(content)

with open('text.txt') as f:
    content = f.readlines()   # This method is used to read all the lines from a text file and return them as a list of strings
print(content)

mylist = ['hello', 'test']
with open('text.txt') as f:
    lines = f.readlines()     # returns a list of string
    mylist = mylist + lines
    print(mylist)


# 2. Writing to a Text File
with open('text.txt', 'w') as f:
    f.write("Hello World") # writes a single string


with open('text.txt', 'a') as f:    # a - appends to file
    f.writelines(mylist)    # Writes a list of strings to the file.(list or any iterable)


# 3. Reading and Writing to a Text file

# r+ for both reading and writing
with open('text.txt', 'r+') as f:
    f.write('Hello World')

    f.seek(0)
    content = f.read()
    print(content)


"""
open(): Opens a file and returns a file object, allowing you to read, write, or append to the file.
close(): Closes the file to ensure that all data is saved and resources are released.
with (context manager): Automatically opens and closes the file, ensuring proper resource management.

read(): Reads the entire file content as a single string or up to a specified number of bytes.
readline(): Reads a single line from the file, returning it as a string.
readlines(): Reads all lines from the file and returns them as a list of strings.

write(): Writes a single string to the file, useful for writing text data.
writelines(): Writes a list of strings to the file, without adding newlines automatically.

seek(): Moves the file pointer to a specified position within the file, useful for random access.
tell(): Returns the current position of the file pointer.
"""


# 4. To check if file/directory is present
import os
p = os.path.isfile('demo.txt')  # to check if file is present at a particular path
p1 = os.path.isdir('demo.txt')  # to check if dir is present at a particular path
dirList = os.listdir('C:\\Windows') # to print all directories in windows

print(p)
print(p1)
print(dirList)


"""
json.load(fp): Reads a JSON object from a file object fp and converts it into a Python dictionary.
json.loads(s): Parses a JSON string s and converts it into a Python dictionary.
json.dump(obj, fp): Converts a Python object obj into a JSON string and writes it to a file object fp.
json.dumps(obj): Converts a Python object obj into a JSON string.
"""