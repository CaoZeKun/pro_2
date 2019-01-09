from functools import wraps


def a_new_decorator(a_func):
    @wraps(a_func)
    def wrapTheFunction():
        print("I am doing some boring work before executing a_func()")
        a_func()
        print("I am doing some boring work after executing a_func()")

    return wrapTheFunction


@a_new_decorator
def a_function_requiring_decoration():
    """Hey yo! Decorate me!"""
    print("I am the function which needs some decoration to "
          "remove my foul smell")

print(a_function_requiring_decoration())
# print(a_function_requiring_decoration.__name__)

import time
def show_time(f):
    def inner():
        start = time.time()
        f()
        end = time.time()
        print('spend %s'%(end - start))
    return inner

@show_time #foo=show_time(f)
def foo():
    print('foo...')
    time.sleep(1)
foo()

def bar1():
    print('bar...')
    time.sleep(2)
# bar1()

@show_time
def bar():
    print('bar...')
    time.sleep(2)
# bar()

def decorate(func):
    print('running decorate', func)
    def decorate_inner():
        print('running decorate_inner function')
        return func()
    return decorate_inner

@decorate
def func_1():
    print('running func_1')
# func_1()
print(func_1)