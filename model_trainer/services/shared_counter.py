from multiprocessing import  Lock


class SharedCounter(object):
    __locker: Lock
    __max_value: int
    __value: int

    def __init__(self, max_value: int):
        self.__value = 0
        self.__max_value = max_value
        self.__locker = Lock()

    def get_max(self):
        return self.__max_value

    def decrement(self, value=1):
        self.__locker.acquire()
        try:
            self.__value -= value
        finally:
            self.__locker.release()

    def try_lock_workers(self, num_workers=1):
        self.__locker.acquire()
        try:
            vacant = self.__max_value - self.__value
            possible = min(vacant, num_workers)
            if possible > 0:
                self.__value += possible
            return possible
        finally:
            self.__locker.release()

    def value(self):
        self.__locker.acquire()
        try:
            return self.__value
        finally:
            self.__locker.release()
