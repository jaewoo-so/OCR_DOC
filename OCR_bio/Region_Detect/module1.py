import inspect

class test:
    def __init__ (self):
        self.val1 = 1
        self.val2 = 100


if __name__ == "__main__":
    t = test()

    templist = [ i for i in dir(t)  ]
    templist2= [ i for i in dir(t) if not callable(i)]

    memberlist = []

    print()


