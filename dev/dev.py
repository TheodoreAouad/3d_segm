
funcs = []
for i in range(4):
    def get_fn(k):
        def fn():
            print(k)
        return fn
    funcs.append(get_fn(i))

i  = 54

for fn in funcs:
    fn()
