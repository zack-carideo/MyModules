import typing 
import time 
import functools 
import logging 
logger = logging.getLogger(__name__)

'''
@Timer
decorated function to time how long it takes any function to execute and print to log 
1. start timer
2. execute the decorated function 
3. stop timer 
'''
def Timer(func: typing.Callable):
    '''
    Timer decorator
    a function to be used to time the execution of other functions

    Params 
    func: a function you want to time 
    '''

    @functools.wraps(func)

    # define the 'inner' function to be executed when the decorator(@Timer) is called(note it takes in 1 input, a function)
    def time_me(*args, **kwargs ) -> typing.Any:

        # create meta data about the function the decorator is applied to
        func_name = func.__name__

        # start the timer
        start = time.time()

        # execute the function being decorated (we use args , kwargs so the decorator can be used to evaluate any function and generalizes)
        value = func(*args, **kwargs)

        # note end time
        end = time.time()

        # write result to log
        logger.info(f'Operation {func_name} took {(end - start):.5f}s.')
        return value

    return time_me