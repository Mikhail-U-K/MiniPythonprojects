def print_hi():
    print("Python is awesome!")
    var = "first string"
    print(var)
    var = 2 + 9
    print(var)


def some_stuff():
    print("Monty Python's flying circus")
    print('Monty Python\'s flying circus')
    print("String")
    print('String')
    print('''Str 
    ... ing''')
    print('Str"ing')


def simple_if(var):
    if var < 3:
        print("less")
    else:
        print("more")


def stdin():
    str1 = input()
    str2 = input()
    print(str1 + ", " + str2)


def none_value():
    null_variable = None
    not_null_variable = 'something'
    if null_variable is None:
        print('null_variable is None')
    else:
        print('null_variable is not None')
    if not_null_variable is None:
        print('not_null_variable is None')
    else:
        print('not_null_variable is not None')


if __name__ == '__main__':
    print_hi()
    some_stuff()
    simple_if(10)
    # stdin()
    none_value()
    print(type(1))
    print(type(5.3))
    print(type(5 + 4j))
    print(type([1, 5.3, False, 4]))
    print(type((1, True, 3, 5 + 4j)))
    print(type(range(5)))
    print(type('Hello'))
    print(type(b'a'))
    print(type(bytearray([1, 2, 3])))
    print(type(memoryview(bytearray('XYZ', 'utf-8'))))
    print(type({'a', 3, True}))
    print(type(frozenset({1, 2, 3})))
    print(type({'a': 32}))