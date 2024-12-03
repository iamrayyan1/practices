print("First")
a = 5
b = 0
d = []

try:
    c = a/b
    print(c)
except ZeroDivisionError:             # If an error occurs, it will print an error message. If error doesn't occur, it will move to the else part
    print("Division by zero")
except EOFError:                      # In this code, it will simply pass and do nothing else (won't move to else part)
    pass
except IndexError:
    print("IndexError")
except Exception as e:               # When error is initially unknown
    print(f"Handle error: {e}")
except:                             # to handle all remaining errors
    print("Unexpected error")

else:
    print(c)

print("last")                      # this will be printed when errors are handled, else the program will crash


