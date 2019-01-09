
# abc(1,2,'lala',[1,2,5],['a','cd'],[])
# abc(1,2)
key = {'a':3,'b':4,'c':2,'d':1}


def abc(l,*args,**kwargs):
    # print(l)

    print(args)
    print(args)
    print(len(args))
    print(kwargs)
    key = kwargs['a']
    print(key)
    # print(args[0][2])
    # print(len(args[0]))
    # if len(args) > 0:
    #     print(args)
    #     print(args[0])
    #     print(len(args))
    #     d = [key[i] for i in args[0]]
    #     print(d)

# abc(1,['a','b','c','d'])
# abc(1,[4,2],2,3)
d = {'a':1,'b':2,'c':3,'d':4}
# args = [2]
# abc(1,args)
print("*" *10)
args = 2,3,[2,45,3]
# abc(1,args,a=d)
abc(1,a=d)