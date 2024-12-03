# 1. Find all
import re

txt = 'The rain in Spain'
x = re.findall('Spain', txt)
print(x)
print(f"Found {len(x)} matches")
# findall: returns a list containing all matches
# returns an empty list if no match was found


# 2. search: Returns a Match object if there is a match anywhere in the string
pattern = r"(\w+)\s(\d+)"
text = "Item 123 is found"

match = re.search(pattern, text)

if match:
    print("Full match:", match.group(0))  # Full match (entire matched string)      Item 123
    print("First group (word):", match.group(1))  # First capture group (word)      Item
    print("Second group (digits):", match.group(2))  # Second capture group (digits) 123
else:
    print("No match found.")
# if no match is found, the value None is returned


# 3. split: Returns a list where the string has been split at each match
import re
txt = 'The rain in Spain'
x = re.split(' ', txt)         # useful for splitting words or sentences
print(x)

# You can control the number of occurrences by specifying the maxsplit parameter
import re
txt = 'The rain in Spain'
x = re.split(' ', txt, 2)
print(x)


# 4. sub: replaces one or many matches with a string
txt = 'The rain in Spain'
x = re.sub('i', 'j', txt)
print(x)


# 5. finditer: returns an iterator that yields Match objects for all non-overlapping matches found anywhere in the string
txt = 'The rain in Spain'
x = re.finditer('ai', txt)

for match in x:
    print(match)



# Match object: object containing information about the search and the result. These methods can be used to retrieve information about the search
#  .span(): Returns a tuple containing the start-, and end positions of the match.
#  .string: Returns the string passed into the function.
#  .group(): Returns the part of the string where there was a match.


"""
re.match(pattern, string): Tries to match the pattern at the start of the string. Returns a match object if found, otherwise None.
re.search(pattern, string): Searches for the first location where the pattern matches in the string. Returns a match object if found, otherwise None.
re.findall(pattern, string): Returns all non-overlapping matches of pattern in the string as a list.
re.finditer(pattern, string): Returns an iterator yielding match objects over all matches of the pattern in the string.
re.sub(pattern, repl, string): Replaces occurrences of the pattern in the string with repl (a replacement string or function).
re.split(pattern, string): Splits the string by occurrences of the pattern. Returns a list of substrings.
re.fullmatch(pattern, string): Tries to match the entire string to the pattern. Returns a match object if the entire string matches.
"""




