
# abc(1,2,'lala',[1,2,5],['a','cd'],[])
# abc(1,2)
key = {'a':3,'b':4,'c':2,'d':1}


def abc(l,*args):
    print(l)
    print(len(args))
    if len(args) > 0:
        print(args)
        print(args[0])
        print(len(args))
        d = [key[i] for i in args[0]]
        print(d)

abc(1,['a','b','c','d'])

